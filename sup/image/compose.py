"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Image Composition Operation Support
"""

import sys
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from blendmodes.blend import BlendType, blendLayers

from loguru import logger

from . import TYPE_IMAGE, TYPE_PIXEL, TYPE_fCOORD2D, \
    image_convert, image_mask, image_mask_add, image_matte, \
    bgr2image, cv2pil, image2bgr, pil2cv

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

class EnumShapes(Enum):
    CIRCLE = 0
    SQUARE = 1
    ELLIPSE = 2
    RECTANGLE = 3
    POLYGON = 4

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

def image_blend(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, mask:Optional[TYPE_IMAGE]=None,
                blendOp:BlendType=BlendType.NORMAL, alpha:float=1) -> TYPE_IMAGE:
    """Blending that will size to the largest input's background."""

    # prep A
    h, w = imageA.shape[:2]
    imageA = image_convert(imageA, 4, w, h)
    imageA = cv2pil(imageA)

    # prep B
    cc = imageB.shape[2] if imageB.ndim > 2 else 1
    imageB = image_convert(imageB, 4, w, h)
    old_mask = image_mask(imageB)

    if mask is None:
        mask = old_mask
    else:
        mask = image_convert(mask, 1, w, h)
        mask = mask[..., 0][:,:]
        if cc == 4:
            mask = cv2.bitwise_and(mask, old_mask)

    imageB[..., 3] = mask
    imageB = cv2pil(imageB)
    alpha = np.clip(alpha, 0, 1)
    image = blendLayers(imageA, imageB, blendOp.value, alpha)
    image = pil2cv(image)
    if cc == 4:
        image = image_mask_add(image, mask)
    return image

def image_crop_polygonal(image: TYPE_IMAGE, points: List[TYPE_fCOORD2D]) -> TYPE_IMAGE:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]
    point_mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    point_mask = cv2.fillPoly(point_mask, [points], 255)
    x, y, w, h = cv2.boundingRect(point_mask)
    cropped_image = cv2.resize(image[y:y+h, x:x+w], (w, h)).astype(np.uint8)
    # Apply the mask to the cropped image
    point_mask_cropped = cv2.resize(point_mask[y:y+h, x:x+w], (w, h))
    if cc == 4:
        mask = image_mask(image, 0)
        alpha_channel = cv2.resize(mask[y:y+h, x:x+w], (w, h))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2BGR)
        cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)
        return image_mask_add(cropped_image, alpha_channel)
    elif cc == 1:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
        cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)
        return image_convert(cropped_image, cc)
    return cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)

def image_crop(image: TYPE_IMAGE, width:int=None, height:int=None, offset:Tuple[float, float]=(0, 0)) -> TYPE_IMAGE:
    h, w = image.shape[:2]
    width = width if width is not None else w
    height = height if height is not None else h
    x, y = offset
    x = max(0, min(width, x))
    y = max(0, min(width, y))
    x2 = max(0, min(width, x + width))
    y2 = max(0, min(height, y + height))
    points = [(x, y), (x2, y), (x2, y2), (x, y2)]
    return image_crop_polygonal(image, points)

def image_crop_center(image: TYPE_IMAGE, width:int=None, height:int=None) -> TYPE_IMAGE:
    """Helper crop function to find the "center" of the area of interest."""
    h, w = image.shape[:2]
    cx = w // 2
    cy = h // 2
    width = w if width is None else width
    height = h if height is None else height
    x1 = max(0, int(cx - width // 2))
    y1 = max(0, int(cy - height // 2))
    x2 = min(w, int(cx + width // 2)) - 1
    y2 = min(h, int(cy + height // 2)) - 1
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return image_crop_polygonal(image, points)

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

def image_split(image: TYPE_IMAGE) -> Tuple[TYPE_IMAGE, ...]:
    h, w = image.shape[:2]

    # Grayscale image
    if image.ndim == 2 or image.shape[2] == 1:
        r = g = b = image
        a = np.full((h, w), 255, dtype=image.dtype)

    # BGR image
    elif image.shape[2] == 3:
        r, g, b = cv2.split(image)
        a = np.full((h, w), 255, dtype=image.dtype)
    else:
        r, g, b, a = cv2.split(image)
    return r, g, b, a

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

# ==============================================================================
# === SHAPE ===
# ==============================================================================

def shape_ellipse(width: int, height: int, sizeX:float=1., sizeY:float=1.,
                  fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    ImageDraw.Draw(image).ellipse(xy, fill=fill)
    return image

def shape_quad(width: int, height: int, sizeX:float=1., sizeY:float=1.,
               fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    ImageDraw.Draw(image).rectangle(xy, fill=fill)
    return image

def shape_polygon(width: int, height: int, size: float=1., sides: int=3,
                  fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), back)
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image
