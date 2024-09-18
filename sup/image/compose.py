"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Image Composition Operation Support
"""

from enum import Enum
import sys
from typing import List, Tuple

import cv2
import numpy as np
from blendmodes.blend import BlendType

from loguru import logger

from Jovimetrix.sup.image import TYPE_IMAGE, TYPE_PIXEL, EnumInterpolation, \
    EnumScaleMode, bgr2image, image2bgr, image_blend, image_convert, \
    image_mask, image_matte, image_minmax

from Jovimetrix.sup.image.adjust import image_scalefit

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumAdjustOP(Enum):
    BLUR = 0
    STACK_BLUR = 1
    GAUSSIAN_BLUR = 2
    MEDIAN_BLUR = 3
    SHARPEN = 10
    EMBOSS = 20
    INVERT = 25
    # MEAN = 30 -- in UNARY
    # ADAPTIVE_HISTOGRAM = 35
    HSV = 30
    LEVELS = 35
    EQUALIZE = 40
    PIXELATE = 50
    QUANTIZE = 55
    POSTERIZE = 60
    FIND_EDGES = 80
    OUTLINE = 70
    DILATE = 71
    ERODE = 72
    OPEN = 73
    CLOSE = 74

class EnumBlendType(Enum):
    """Rename the blend type names."""
    NORMAL = BlendType.NORMAL
    ADDITIVE = BlendType.ADDITIVE
    NEGATION = BlendType.NEGATION
    DIFFERENCE = BlendType.DIFFERENCE
    MULTIPLY = BlendType.MULTIPLY
    DIVIDE = BlendType.DIVIDE
    LIGHTEN = BlendType.LIGHTEN
    DARKEN = BlendType.DARKEN
    SCREEN = BlendType.SCREEN
    BURN = BlendType.COLOURBURN
    DODGE = BlendType.COLOURDODGE
    OVERLAY = BlendType.OVERLAY
    HUE = BlendType.HUE
    SATURATION = BlendType.SATURATION
    LUMINOSITY = BlendType.LUMINOSITY
    COLOR = BlendType.COLOUR
    SOFT = BlendType.SOFTLIGHT
    HARD = BlendType.HARDLIGHT
    PIN = BlendType.PINLIGHT
    VIVID = BlendType.VIVIDLIGHT
    EXCLUSION = BlendType.EXCLUSION
    REFLECT = BlendType.REFLECT
    GLOW = BlendType.GLOW
    XOR = BlendType.XOR
    EXTRACT = BlendType.GRAINEXTRACT
    MERGE = BlendType.GRAINMERGE
    DESTIN = BlendType.DESTIN
    DESTOUT = BlendType.DESTOUT
    SRCATOP = BlendType.SRCATOP
    DESTATOP = BlendType.DESTATOP

class EnumImageBySize(Enum):
    LARGEST = 10
    SMALLEST = 20
    WIDTH_MIN = 30
    WIDTH_MAX = 40
    HEIGHT_MIN = 50
    HEIGHT_MAX = 60

class EnumOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    GRID = 2

# ==============================================================================
# === PIXEL ===
# ==============================================================================

def pixel_convert(color:TYPE_PIXEL, size:int=4, alpha:int=255) -> TYPE_PIXEL:
    """Convert X channel pixel into Y channel pixel."""
    if (cc := len(color)) == size:
        return color
    if size > 2:
        color += (0,) * (3 - cc)
        if size == 4:
            color += (alpha,)
        return color
    return color[0]

# ==============================================================================
# === IMAGE ===
# ==============================================================================
"""
These are core functions that most of the support image libraries require.
"""

def image_flatten(image: List[TYPE_IMAGE], width:int=None, height:int=None,
                  mode=EnumScaleMode.MATTE,
                  sample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:

    if mode == EnumScaleMode.MATTE:
        width, height, _, _ = image_minmax(image)[1:]
    else:
        h, w = image[0].shape[:2]
        width = width or w
        height = height or h

    current = np.zeros((height, width, 4), dtype=np.uint8)
    for x in image:
        if mode != EnumScaleMode.MATTE:
            x = image_scalefit(x, width, height, mode, sample)
        x = image_matte(x, (0,0,0,0), width, height)
        x = image_scalefit(x, width, height, EnumScaleMode.CROP, sample)
        x = image_convert(x, 4)
        #@TODO: ADD VARIOUS COMP OPS?
        current = cv2.add(current, x)
    return current

def image_flatten_mask(image:TYPE_IMAGE, matte:Tuple=(0,0,0,255)) -> Tuple[TYPE_IMAGE, TYPE_IMAGE|None]:
    """Flatten the image with its own alpha channel, if any."""
    mask = image_mask(image)
    return image_blend(image, image, mask), mask

def image_levels(image: np.ndarray, black_point:int=0, white_point=255,
        mid_point=128, gamma=1.0) -> np.ndarray:
    """
    Adjusts the levels of an image including black, white, midpoints, and gamma correction.

    Args:
        image (numpy.ndarray): Input image tensor in RGB(A) format.
        black_point (int): The black point to adjust shadows. Default is 0.
        white_point (int): The white point to adjust highlights. Default is 255.
        mid_point (int): The mid point for mid-tone adjustment. Default is 128.
        gamma (float): Gamma correction value. Default is 1.0.

    Returns:
        numpy.ndarray: Adjusted image tensor.
    """

    image, alpha, cc = image2bgr(image)

    # Convert points and gamma to float32 for calculations
    black = np.array([black_point] * 3, dtype=np.float32)
    white = np.array([white_point] * 3, dtype=np.float32)
    mid = np.array([mid_point] * 3, dtype=np.float32)
    inGamma = np.array([gamma] * 3, dtype=np.float32)
    outBlack = np.array([0, 0, 0], dtype=np.float32)
    outWhite = np.array([255, 255, 255], dtype=np.float32)

    # Apply levels adjustment
    image = np.clip((image - black) / (white - black), 0, 1)
    image = (image - mid) / (1.0 - mid)
    image = (image ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    image = np.clip(image, 0, 255).astype(np.uint8)
    return bgr2image(image, alpha, cc == 1)

def image_mask_binary(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """
    Convert an image to a binary mask where non-black pixels are 1 and black pixels are 0.
    Supports BGR, single-channel grayscale, and RGBA images.

    Args:
        image (TYPE_IMAGE): Input image in BGR, grayscale, or RGBA format.

    Returns:
        TYPE_IMAGE: Binary mask with the same width and height as the input image, where
                    pixels are 1 for non-black and 0 for black.
    """
    if image.ndim == 2:
        # Grayscale image
        gray = image
    elif image.shape[2] == 3:
        # BGR image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        # RGBA image
        alpha_channel = image[..., 3]
        # Create a mask from the alpha channel where alpha > 0
        alpha_mask = alpha_channel > 0
        # Convert RGB to grayscale
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        # Apply the alpha mask to the grayscale image
        gray = cv2.bitwise_and(gray, gray, mask=alpha_mask.astype(np.uint8))
    else:
        raise ValueError("Unsupported image format")

    # Create a binary mask where any non-black pixel is set to 1
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)
    return mask.astype(np.uint8)

def image_by_size(image_list: List[TYPE_IMAGE],
                  enumSize: EnumImageBySize=EnumImageBySize.LARGEST) -> Tuple[TYPE_IMAGE, int, int]:

    img = None
    mega, width, height = 0, 0, 0
    if enumSize in [EnumImageBySize.SMALLEST, EnumImageBySize.WIDTH_MIN, EnumImageBySize.HEIGHT_MIN]:
        mega, width, height = sys.maxsize, sys.maxsize, sys.maxsize

    for i in image_list:
        h, w = i.shape[:2]
        match enumSize:
            case EnumImageBySize.LARGEST:
                if (new_mega := w * h) > mega:
                    mega = new_mega
                    img = i
                width = max(width, w)
                height = max(height, h)
            case EnumImageBySize.SMALLEST:
                if (new_mega := w * h) < mega:
                    mega = new_mega
                    img = i
                width = min(width, w)
                height = min(height, h)
            case EnumImageBySize.WIDTH_MIN:
                if w < width:
                    width = w
                    img = i
            case EnumImageBySize.WIDTH_MAX:
                if w > width:
                    width = w
                    img = i
            case EnumImageBySize.HEIGHT_MIN:
                if h < height:
                    height = h
                    img = i
            case EnumImageBySize.HEIGHT_MAX:
                if h > height:
                    height = h
                    img = i

    return img, width, height

def image_stack(image_list: List[TYPE_IMAGE],
                axis:EnumOrientation=EnumOrientation.HORIZONTAL,
                stride:int=0, matte:TYPE_PIXEL=(0,0,0,255)) -> TYPE_IMAGE:

    _, width, height = image_by_size(image_list)
    images = [image_matte(image_convert(i, 4), matte, width, height) for i in image_list]
    count = len(images)

    matte = pixel_convert(matte, 4)
    match axis:
        case EnumOrientation.GRID:
            if stride < 1:
                stride = np.ceil(np.sqrt(count))
                stride = int(stride)
            stride = min(stride, count)
            stride = max(stride, 1)

            rows = []
            for i in range(0, count, stride):
                row = images[i:i + stride]
                row_stacked = np.hstack(row)
                rows.append(row_stacked)

            height, width = images[0].shape[:2]
            overhang = count % stride
            if overhang != 0:
                overhang = stride - overhang
                size = (height, overhang * width, 4)
                filler = np.full(size, matte, dtype=np.uint8)
                rows[-1] = np.hstack([rows[-1], filler])
            image = np.vstack(rows)

        case EnumOrientation.HORIZONTAL:
            image = np.hstack(images)

        case EnumOrientation.VERTICAL:
            image = np.vstack(images)
    return image
