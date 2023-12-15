"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix

                    Copyright 2023 Alexander Morano (Joviex)
"""

import math
from enum import Enum
from typing import Any, Optional

import cv2
import torch
import numpy as np
from skimage import exposure
from scipy.ndimage import rotate
from blendmodes.blend import blendLayers, BlendType
from PIL import Image, ImageDraw

from Jovimetrix import Logger, grid_make, cv2mask, cv2tensor, cv2pil, pil2cv,\
    TYPE_IMAGE, TYPE_PIXEL, TYPE_COORD

HALFPI = math.pi / 2
TAU = math.pi * 2

# =============================================================================
# === ENUM GLOBALS ===
# =============================================================================

class EnumGrayscaleCrunch(Enum):
    LOW = 0
    HIGH = 1
    MEAN = 2

class EnumIntFloat(Enum):
    FLOAT = 0
    INT = 1

class EnumImageType(Enum):
    GRAYSCALE = 0
    RGB = 1
    RGBA = 2

class EnumScaleMode(Enum):
    NONE = 0
    FIT = 1
    CROP = 2
    ASPECT = 3

class EnumOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    GRID = 2

class EnumThreshold(Enum):
    BINARY = cv2.THRESH_BINARY
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO

class EnumThresholdAdapt(Enum):
    ADAPT_NONE = -1
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADAPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

class EnumColorMap(Enum):
    AUTUMN = cv2.COLORMAP_AUTUMN
    BONE = cv2.COLORMAP_BONE
    JET = cv2.COLORMAP_JET
    WINTER = cv2.COLORMAP_WINTER
    RAINBOW = cv2.COLORMAP_RAINBOW
    OCEAN = cv2.COLORMAP_OCEAN
    SUMMER = cv2.COLORMAP_SUMMER
    SPRING = cv2.COLORMAP_SPRING
    COOL = cv2.COLORMAP_COOL
    HSV = cv2.COLORMAP_HSV
    PINK = cv2.COLORMAP_PINK
    HOT = cv2.COLORMAP_HOT
    PARULA = cv2.COLORMAP_PARULA
    MAGMA = cv2.COLORMAP_MAGMA
    INFERNO = cv2.COLORMAP_INFERNO
    PLASMA = cv2.COLORMAP_PLASMA
    VIRIDIS = cv2.COLORMAP_VIRIDIS
    CIVIDIS = cv2.COLORMAP_CIVIDIS
    TWILIGHT = cv2.COLORMAP_TWILIGHT
    TWILIGHT_SHIFTED = cv2.COLORMAP_TWILIGHT_SHIFTED
    TURBO = cv2.COLORMAP_TURBO
    DEEPGREEN = cv2.COLORMAP_DEEPGREEN

class EnumInterpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4
    LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
    INTER_MAX = cv2.INTER_MAX
    WARP_FILL_OUTLIERS = cv2.WARP_FILL_OUTLIERS
    WARP_INVERSE_MAP = cv2.WARP_INVERSE_MAP

class EnumAdjustOP(Enum):
    BLUR = 0
    STACK_BLUR = 1
    GAUSSIAN_BLUR = 2
    MEDIAN_BLUR = 3
    SHARPEN = 10
    EMBOSS = 20
    MEAN = 30
    ADAPTIVE_HISTOGRAM = 35
    EQUALIZE = 40
    PIXELATE = 50
    QUANTIZE = 55
    POSTERIZE = 60
    OUTLINE = 70
    DILATE = 71
    ERODE = 72
    OPEN = 73
    CLOSE = 74

class EnumColorTheory(Enum):
    COMPLIMENTARY = 0
    MONOCHROMATIC = 1
    SPLIT_COMPLIMENTARY = 2
    ANALOGOUS = 3
    TRIADIC = 4
    TETRADIC = 5
    SQUARE = 6
    RECTANGULAR = 7
    COMPOUND = 8

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

# =============================================================================
# === NODE SUPPORT ===
# =============================================================================

IT_SAMPLE = {
    "optional": {
        "resample": (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
    }}

# =============================================================================
# === UTILITY ===
# =============================================================================

def gray_sized(image: TYPE_IMAGE, h:int, w:int, resample: EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:
    """Force an image into Grayscale at a specific width, height."""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[0] != h or image.shape[1] != w:
        image = cv2.resize(image, (w, h), interpolation=resample.value)
    return image

def pixel_split(image: TYPE_IMAGE) -> tuple[list[TYPE_IMAGE], list[TYPE_IMAGE]]:
    w, h, c = image.shape

    # Check if the image has an alpha channel
    if c == 4:
        b, g, r, _ = cv2.split(image)
    else:
        b, g, r = cv2.split(image)
        # a = np.zeros_like(r)

    e = np.zeros((h, w), dtype=np.uint8)

    masks = []
    images = []
    f = cv2.merge([r, r, r])
    masks.append(cv2mask(f))

    f = cv2.merge([e, e, r])
    images.append(cv2tensor(f))

    f = cv2.merge([g, g, g])
    masks.append(cv2mask(f))
    f = cv2.merge([e, g, e])
    images.append(cv2tensor(f))

    f = cv2.merge([b, b, b])
    masks.append(cv2mask(f))
    f = cv2.merge([b, e, e])
    images.append(cv2tensor(f))

    return images, masks

def merge_channel(channel, size, resample: EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:
    if channel is None:
        return np.full(size, 0, dtype=np.uint8)
    return gray_sized(channel, *size[::-1], resample)

def image_merge(r: TYPE_IMAGE, g: TYPE_IMAGE, b: TYPE_IMAGE, a: TYPE_IMAGE,
          width: int, height: int,
          mode:EnumScaleMode=EnumScaleMode.NONE,
          resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:

    thr, twr = (r.shape[0], r.shape[1]) if r is not None else (height, width)
    thg, twg = (g.shape[0], g.shape[1]) if g is not None else (height, width)
    thb, twb = (b.shape[0], b.shape[1]) if b is not None else (height, width)
    w = max(width, max(twb, max(twr, twg)))
    h = max(height, max(thb, max(thr, thg)))

    if a is None:
        a = np.full((height, width), 255, dtype=np.uint8)
    else:
        w = max(w, a.shape[1])
        h = max(h, a.shape[0])

    target_size = (w, h)

    r = merge_channel(r, target_size, resample)
    g = merge_channel(g, target_size, resample)
    b = merge_channel(b, target_size, resample)
    a = merge_channel(a, target_size, resample)

    image = cv2.merge((r, g, b))
    return geo_scalefit(image, width, height, mode, resample)

# =============================================================================
# === SHAPE FUNCTIONS ===
# =============================================================================

def shape_body(func: str, width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=1.) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    func = getattr(d, func)
    func(xy, fill=color_eval(fill))
    return image

def shape_ellipse(width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=1.) -> Image:
    return shape_body('ellipse', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

def shape_quad(width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=1.) -> Image:
    return shape_body('rectangle', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

def shape_polygon(width: int, height: int, size: float=1., sides: int=3, angle: float=0., fill:TYPE_PIXEL=1.) -> Image:

    fill = color_eval(fill)
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image

# =============================================================================
# === CHANNEL FUNCTIONS ===
# =============================================================================

def channel_count(image: TYPE_IMAGE) -> tuple[int, EnumImageType]:
    size = image.shape[2] if len(image.shape) > 2 else 1
    mode = EnumImageType.RGBA if size == 4 else EnumImageType.RGB if size == 3 else EnumImageType.GRAYSCALE
    return size, mode

def channel_add(image: TYPE_IMAGE, value: TYPE_PIXEL=1.) -> TYPE_IMAGE:
    new = channel_solid(color=value, image=image)
    return np.concatenate([image, new], axis=-1)

def channel_solid(width: int=512,
                  height:int=512,
                  color:TYPE_PIXEL=1.,
                  image:Optional[TYPE_IMAGE]=None,
                  chan:EnumImageType=EnumImageType.GRAYSCALE) -> TYPE_IMAGE:

    if image is not None:
        height, width = image.shape[:2]

    color = np.asarray(color_eval(color, chan))
    match chan:
        case EnumImageType.GRAYSCALE:
            return np.full((height, width, 1), color, dtype=np.uint8)

        case EnumImageType.RGB:
            return np.full((height, width, 3), color, dtype=np.uint8)

        case EnumImageType.RGBA:
            return np.full((height, width, 4), color, dtype=np.uint8)

# =============================================================================
# === IMAGE FUNCTIONS ===
# =============================================================================

def image_stack(images: list[TYPE_IMAGE],
                axis:EnumOrientation=EnumOrientation.HORIZONTAL,
                stride:Optional[int]=None,
                color:TYPE_PIXEL=0.,
                mode:EnumScaleMode=EnumScaleMode.NONE,
                resample:Image.Resampling=Image.Resampling.LANCZOS) -> TYPE_IMAGE:

    color = (
        (color[0] if color is not None else 1) * 255,
        (color[1] if color is not None else 1) * 255,
        (color[2] if color is not None else 1) * 255
    )

    count = len(images)

    # CROP ALL THE IMAGES TO THE LARGEST ONE OF THE INPUT SET
    width, height = 0, 0
    for i in images:
        w, h = i.shape[:2]
        width = max(width, w)
        height = max(height, h)

    # center = (width // 2, height // 2)
    images = [comp_fill(i, width, height, color, mode, resample) for i in images]

    match axis:
        case EnumOrientation.GRID:
            if not stride:
                stride = int(np.ceil(np.sqrt(count)))

            rows = []
            for i in range(0, count, stride):
                row = images[i:i + stride]
                row_stacked = np.hstack(row)
                rows.append(row_stacked)

            height, width = images[0].shape[:2]
            # Check if the last row needs padding
            overhang = len(images) % stride

            Logger.debug('image_stack', overhang, width, height, )

            if overhang != 0:
                overhang = stride - overhang

                chan = 1
                if len(rows[0].shape) > 2:
                    chan = 3

                size = (height, overhang * width, chan)
                filler = np.full(size, color, dtype=np.uint8)
                rows[-1] = np.hstack([rows[-1], filler])

            image = np.vstack(rows)

        case EnumOrientation.HORIZONTAL:
            image = np.hstack(images)

        case EnumOrientation.VERTICAL:
            image = np.vstack(images)

        case _:
               raise ValueError("image_stack", f"invalid orientation - {axis}")

    return image

def image_grid(data: list[TYPE_IMAGE], width: int, height: int) -> TYPE_IMAGE:
    #@TODO: makes poor assumption all images are the same dimensions.
    chunks, col, row = grid_make(data)
    frame = np.zeros((height * row, width * col, 3), dtype=np.uint8)
    i = 0
    for y, strip in enumerate(chunks):
        for x, item in enumerate(strip):
            y1, y2 = y * height, (y+1) * height
            x1, x2 = x * width, (x+1) * width
            frame[y1:y2, x1:x2, ] = item
            i += 1

    return frame

# =============================================================================
# === GEOMETRY FUNCTIONS ===
# =============================================================================

def geo_crop(image: TYPE_IMAGE,
             left: int=None, top: int=None, right: int=None, bottom: int=None,
             widthT: int=None, heightT: int=None, pad:bool=False,
             color: TYPE_PIXEL=0) -> TYPE_IMAGE:

        height, width, _ = image.shape
        left = float(np.clip(left or 0, 0, 1))
        top = float(np.clip(top or 0, 0, 1))
        right = float(np.clip(right or 1, 0, 1))
        bottom = float(np.clip(bottom or 1, 0, 1))

        if top > bottom:
             bottom, top = top, bottom

        if left > right:
             right, left = left, right

        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2 = width * (right - left)
        ch2 = height * (bottom - top)

        crop_img = image[max(0, mid_y - ch2):min(mid_y + ch2, height),
                         max(0, mid_x - cw2):min(mid_x + cw2, width)]

        widthT = (widthT if widthT is not None else width)
        heightT = (heightT if heightT is not None else height)
        if (widthT == width and heightT == height) or not pad:
            return crop_img

        cc, _ = channel_count(image)
        img_padded = np.full((heightT, widthT, cc), color, dtype=np.uint8)

        crop_height, crop_width, _ = crop_img.shape
        h2 = heightT // 2
        w2 = widthT // 2
        ch = crop_height // 2
        cw = crop_width // 2
        y_start, y_end = max(0, h2 - ch), min(h2 + ch, heightT)
        x_start, x_end = max(0, w2 - cw), min(w2 + cw, widthT)

        y_delta = (y_end - y_start) // 2
        x_delta = (x_end - x_start) // 2
        y_start2, y_end2 = max(0, ch - y_delta), min(ch + y_delta, crop_height)
        x_start2, x_end2 = max(0, cw - x_delta), min(cw + x_delta, crop_width)

        img_padded[y_start:y_end, x_start:x_end] = crop_img[y_start2:y_end2, x_start2:x_end2]
        # Logger.debug("geo_crop", f"({x_start}, {y_start})-({x_end}, {y_end}) || ({x_start2}, {y_start2})-({x_end2}, {y_end2})")
        return img_padded

def geo_edge_wrap(image: TYPE_IMAGE, tileX: float=1., tileY: float=1., edge: str='WRAP') -> TYPE_IMAGE:
    """TILING."""
    height, width, _ = image.shape
    tileX = int(tileX * width * 0.5) if edge in ["WRAP", "WRAPX"] else 0
    tileY = int(tileY * height * 0.5) if edge in ["WRAP", "WRAPY"] else 0
    # Logger.debug("geo_edge_wrap", f"[{width}, {height}]  [{tileX}, {tileY}]")
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

def geo_translate(image: TYPE_IMAGE, offsetX: float, offsetY: float) -> TYPE_IMAGE:
    """TRANSLATION."""
    height, width, _ = image.shape
    M = np.float32([[1, 0, offsetX * width], [0, 1, offsetY * height]])
    # Logger.debug("geo_translate", f"[{offsetX}, {offsetY}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def geo_rotate(image: TYPE_IMAGE, angle: float, center:TYPE_COORD=(0.5 ,0.5)) -> TYPE_IMAGE:
    """ROTATION."""
    height, width, _ = image.shape
    center = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # Logger.debug("geo_rotate", f"[{angle}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def geo_rotate_array(image: TYPE_IMAGE, angle: float, clip: bool=True) -> TYPE_IMAGE:
    """."""
    rotated_image = rotate(image, angle, reshape=not clip, mode='constant', cval=0)

    if not clip:
        return rotated_image

    # Compute the dimensions for clipping
    height, width, _ = image.shape
    rotated_height, rotated_width, _ = rotated_image.shape

    # Calculate the difference in dimensions
    height_diff = rotated_height - height
    width_diff = rotated_width - width

    # Calculate the starting indices for clipping
    start_height = height_diff // 2
    start_width = width_diff // 2

    # Clip the rotated image
    return rotated_image[start_height:start_height + height, start_width:start_width + width]

def geo_scalefit(image: TYPE_IMAGE,
                 width: int, height:int,
                 mode:EnumScaleMode=EnumScaleMode.NONE,
                 resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:

    Logger.spam(mode, width, height, resample)

    match mode:
        case EnumScaleMode.ASPECT:
            h, w = image.shape[:2]
            aspect = min(width / w, height / h)
            return cv2.resize(image, None, fx=aspect, fy=aspect, interpolation=resample.value)

        case EnumScaleMode.CROP:
            return geo_crop(image, widthT=width, heightT=height, pad=True)

        case EnumScaleMode.FIT:
            return cv2.resize(image, (width, height), interpolation=resample.value)

    return image

def geo_transform(image: TYPE_IMAGE, offsetX: float=0., offsetY: float=0., angle: float=0.,
              sizeX: float=1., sizeY: float=1., edge:str='CLIP', widthT: int=256, heightT: int=256,
              mode:EnumScaleMode=EnumScaleMode.NONE,
              resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:
    """Transform, Rotate and Scale followed by Tiling and then Inversion, conforming to an input wT, hT,."""

    height, width, _ = image.shape

    # SCALE
    if sizeX != 1. or sizeY != 1.:
        wx = int(width * sizeX)
        hx = int(height * sizeY)
        image = cv2.resize(image, (wx, hx), interpolation=resample.value)

    # ROTATION
    if angle != 0:
        image = geo_rotate(image, angle)

    # TRANSLATION
    if offsetX != 0. or offsetY != 0.:
        image = geo_translate(image, offsetX, offsetY)

    if edge != "CLIP":
        tx = ty = 0
        if edge in ["WRAP", "WRAPX"] and sizeX < 1.:
            tx = 1. / sizeX - 1
            sizeX = 1.

        if edge in ["WRAP", "WRAPY"] and sizeY < 1.:
            ty = 1. / sizeY - 1
            sizeY = 1.

        image = geo_edge_wrap(image, tx, ty)
        h, w, _ = image.shape

    # clip to original size first...
    image = geo_crop(image)
    # Logger.debug("geo_transform", f"({offsetX},{offsetY}), {angle}, ({sizeX},{sizeY}) [{width}x{height} - {mode} - {resample}]")
    return geo_scalefit(image, widthT, heightT, mode, resample)

def geo_merge(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, axis: int=0, flip: bool=False) -> TYPE_IMAGE:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def geo_mirror(image: TYPE_IMAGE, pX: float, axis: int, invert: bool=False) -> TYPE_IMAGE:
    output =  np.zeros_like(image)
    flip = cv2.flip(image, axis)
    height, width, _ = image.shape

    pX = np.clip(pX, 0, 1)
    if invert:
        pX = 1 - pX
        flip, image = image, flip

    scalar = height if axis == 0 else width
    slice1 = int(pX * scalar)
    slice1w = scalar - slice1
    slice2w = min(scalar - slice1w, slice1w)

    if axis == 0:
        output[:slice1, :] = image[:slice1, :]
        output[slice1:slice1 + slice2w, :] = flip[slice1w:slice1w + slice2w, :]
    else:
        output[:, :slice1] = image[:, :slice1]
        output[:, slice1:slice1 + slice2w] = flip[:, slice1w:slice1w + slice2w]

    if invert:
        output = cv2.flip(output, axis)

    return output

# =============================================================================
# === LIGHT FUNCTIONS ===
# =============================================================================

def light_hsv(image: TYPE_IMAGE, hue: float, saturation: float, value: float) -> TYPE_IMAGE:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue *= 255
    image[:, :, 0] = (image[:, :, 0] + hue) % 180
    image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
    # Logger.debug("light_hsv", f"{hue} {saturation} {value}")
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def light_gamma(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    # Logger.debug("light_gamma", f"({value})")
    if value == 0:
        return (image * 0).astype(np.uint8)

    invGamma = 1.0 / max(0.000001, value)
    lookUpTable = np.clip(np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype(np.uint8), 0, 255)
    return cv2.LUT(image, lookUpTable)

def light_contrast(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    # Logger.debug("light_contrast", f"({value})")
    mean_value = np.mean(image)
    image = (image - mean_value) * value + mean_value
    return np.clip(image, 0, 255).astype(np.uint8)

def light_exposure(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    # Logger.debug("light_exposure", f"({math.pow(2.0, value)})")
    return np.clip(image * value, 0, 255).astype(np.uint8)

def light_invert(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    value = np.clip(value, 0, 1)
    return cv2.addWeighted(image, 1 - value, 255 - image, value, 0)

# =============================================================================
# === COLOR FUNCTIONS ===
# =============================================================================

def color_eval(color: TYPE_PIXEL,
               mode: EnumImageType=EnumImageType.RGB,
               target:EnumIntFloat=EnumIntFloat.INT,
               crunch:EnumGrayscaleCrunch=EnumGrayscaleCrunch.MEAN) -> TYPE_PIXEL:

    def parse_single_color(c: TYPE_PIXEL) -> TYPE_PIXEL:
        if isinstance(c, float) or c != int(c):
            c = max(0, min(1, c))
            if target == EnumIntFloat.INT:
                c = int(c * 255)

        elif isinstance(c, int):
            c = max(0, min(255, c))
            if target == EnumIntFloat.FLOAT:
                c /= 255.
        return c

    Logger.spam(color, mode, target, crunch)
    if mode == EnumImageType.GRAYSCALE:
        if isinstance(color, (int, float)):
            return parse_single_color(color)
        elif isinstance(color, (set, tuple, list)):
            data = [parse_single_color(x) for x in color]
            match crunch:
                case EnumGrayscaleCrunch.LOW:
                    return min(data)
                case EnumGrayscaleCrunch.HIGH:
                    return max(data)
                case EnumGrayscaleCrunch.MEAN:
                    return int(np.mean(data))

    elif mode == EnumImageType.RGB:
        if isinstance(color, (int, float)):
            c = parse_single_color(color)
            return (c, c, c)

        elif isinstance(color, (set, tuple, list)):
            if len(color) < 3:
                a = 255 if target == EnumIntFloat.INT else 1
                color += (a,) * (3 - len(color))
            return color[::-1]

    elif mode == EnumImageType.RGBA:
        if isinstance(color, (int, float)):
            c = parse_single_color(color)
            return (c, c, c, 255) if target == EnumIntFloat.INT else (c, c, c, 1)

        elif isinstance(color, (set, tuple, list)):
            if len(color) < 4:
                a = 255 if target == EnumIntFloat.INT else 1
                color += (a,) * (4 - len(color))
            return (color[2], color[1], color[0], color[3])

    return color[::-1]

def color_lut_from_image(image: TYPE_IMAGE, num_colors:int=256) -> TYPE_IMAGE:
    """Create X sized LUT from an RGB image."""
    image = cv2.resize(image, (num_colors, 1))
    return image.reshape(-1, 3).astype(np.uint8)

def color_match(image: TYPE_IMAGE, usermap: TYPE_IMAGE) -> TYPE_IMAGE:
    """Colorize one input based on the histogram matches."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    beta = cv2.cvtColor(usermap, cv2.COLOR_BGR2LAB)
    image = exposure.match_histograms(image, beta, channel_axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return comp_blend(usermap, image, blendOp=BlendType.LUMINOSITY)

def color_match_reinhard(image: TYPE_IMAGE, target: TYPE_IMAGE) -> TYPE_IMAGE:
    """Reinhard Color matching based on https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer."""
    lab_tar = cv2.cvtColor(target, cv2.COLOR_BGR2Lab)
    lab_ori = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    mean_tar, std_tar = cv2.meanStdDev(lab_tar)
    mean_ori, std_ori = cv2.meanStdDev(lab_ori)
    ratio = (std_tar/std_ori).reshape(-1)
    offset = (mean_tar - mean_ori*std_tar/std_ori).reshape(-1)
    lab_tar = cv2.convertScaleAbs(lab_ori*ratio + offset)
    return cv2.cvtColor(lab_tar, cv2.COLOR_Lab2BGR)

def color_match_custom_map(image: TYPE_IMAGE,
                   usermap: Optional[TYPE_IMAGE]=None,
                   colormap: int=cv2.COLORMAP_JET) -> TYPE_IMAGE:
    """Colorize one input based on custom GNU Octave/MATLAB map"""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image[:, :, 1]
    if usermap is not None:
        usermap = color_lut_from_image(usermap)
        return cv2.applyColorMap(image, usermap)
    return cv2.applyColorMap(image, colormap)

def color_match_heat_map(image: TYPE_IMAGE,
                  threshold:float=0.55,
                  colormap:int=cv2.COLORMAP_JET,
                  sigma:int=13) -> TYPE_IMAGE:
    """Colorize one input based on custom GNU Octave/MATLAB map"""

    threshold = min(1, max(0, threshold)) * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image[:, :, 1]
    image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

    sigma = max(3, sigma)
    if sigma % 2 == 0:
        sigma += 1
    sigmaY = sigma - 2

    image = cv2.GaussianBlur(image, (sigma, sigma), sigmaY)
    image = cv2.applyColorMap(image, colormap)
    return cv2.addWeighted(image, 0.5, image, 0.5, 0)

def color_average(image: TYPE_IMAGE) -> TYPE_IMAGE:
    return np.mean(image)
    return adjust_pixelate(image, 256)

def color_theory_complementary(color: TYPE_PIXEL) -> TYPE_PIXEL:
    return tuple(max(0, min(255, 255 - c)) for c in color)

def color_theory_monochromatic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    return [
        color_theory_complementary((color[0] + 30, color[1] + 30, color[2] + 30)),
        color_theory_complementary((color[0] - 30, color[1] - 30, color[2] - 30))
    ]

def color_theory_split_complementary(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    return [
        color_theory_complementary((color[0] + 30, color[1] - 30, color[2])),
        color_theory_complementary((color[0] - 30, color[1] + 30, color[2]))
    ]

def color_theory_analogous(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    return [
        color_theory_complementary((color[0] + 20, color[1] + 20, color[2])),
        color_theory_complementary((color[0] - 20, color[1] - 20, color[2]))
    ]

def color_theory_triadic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    return [
        color_theory_complementary((color[0] + 120, color[1] + 120, color[2])),
        color_theory_complementary((color[0] - 120, color[1] - 120, color[2]))
    ]

def color_theory_tetradic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    return [
        color_theory_complementary((color[0] + 60, color[1] + 60, color[2] + 60)),
        color_theory_monochromatic(color)[0],
        color_theory_monochromatic(color)[1]
    ]

def color_theory_square(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    return [
        color_theory_complementary((color[0] + 90, color[1] + 90, color[2])),
        color_theory_complementary((color[0] + 180, color[1] + 180, color[2])),
        color_theory_complementary((color[0] + 270, color[1] + 270, color[2]))
    ]

def color_theory_rectangular(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    return [
        color_theory_complementary((color[0] + 60, color[1] + 60, color[2])),
        color_theory_complementary((color[0] + 180, color[1] + 180, color[2])),
        color_theory_complementary((color[0] + 240, color[1] + 240, color[2]))
    ]

def color_theory_compound(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    return [
        color_theory_complementary((color[0] + 180, color[1] + 180, color[2])),
        color_theory_complementary((color[0] + 30, color[1] + 30, color[2])),
        color_theory_complementary((color[0] + 210, color[1] + 210, color[2]))
    ]

def color_theory(image: TYPE_IMAGE, scheme: EnumColorTheory=EnumColorTheory.COMPLIMENTARY) -> tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE]:

    aR = aG = aB = bR = bG = bB = cR = cG = cB = 0
    image_avg = color_average(image)
    color = image_avg[image_avg.shape[0] // 2, image_avg.shape[1] // 2]
    match scheme:
        case EnumColorTheory.COMPLIMENTARY:
            a = color_theory_complementary(color)
            aR, aG, aB = a
        case EnumColorTheory.MONOCHROMATIC:
            a, b = color_theory_monochromatic(color)
            aR, aG, aB = a
            bR, bG, bB = b
        case EnumColorTheory.SPLIT_COMPLIMENTARY:
            a, b = color_theory_split_complementary(color)
            aR, aG, aB = a
            bR, bG, bB = b
        case EnumColorTheory.ANALOGOUS:
            a, b = color_theory_analogous(color)
            aR, aG, aB = a
            bR, bG, bB = b
        case EnumColorTheory.TRIADIC:
            a, b = color_theory_triadic(color)
            aR, aG, aB = a
            bR, bG, bB = b
        case EnumColorTheory.TETRADIC:
            a, b, c = color_theory_tetradic(color)
            aR, aG, aB = a
            bR, bG, bB = b
            cR, cG, cB = c
        case EnumColorTheory.SQUARE:
            a, b, c = color_theory_square(color)
            aR, aG, aB = a
            bR, bG, bB = b
            cR, cG, cB = c
        case EnumColorTheory.RECTANGULAR:
            a, b, c = color_theory_rectangular(color)
            aR, aG, aB = a
            bR, bG, bB = b
            cR, cG, cB = c
        case EnumColorTheory.COMPOUND:
            a, b, c = color_theory_compound(color)
            aR, aG, aB = a
            bR, bG, bB = b
            cR, cG, cB = c

    return (
        np.full_like(image, [aR, aG, aB], dtype=np.uint8),
        np.full_like(image, [bR, bG, bB], dtype=np.uint8),
        np.full_like(image, [cR, cG, cB], dtype=np.uint8),
        image_avg
    )

# =============================================================================
# === COMP FUNCTIONS ===
# =============================================================================

def comp_lerp(imageA:TYPE_IMAGE,
              imageB:TYPE_IMAGE,
              mask:TYPE_IMAGE=None,
              alpha:float=1.) -> TYPE_IMAGE:

    imageA = imageA.astype(np.float32)
    imageB = imageB.astype(np.float32)

    # normalize alpha and establish mask
    alpha = np.clip(alpha, 0, 1)
    if mask is None:
        height, width, _ = imageA.shape
        mask = cv2.empty((height, width, 1), dtype=cv2.uint8)
    else:
        # normalize the mask
        info = np.iinfo(mask.dtype)
        mask = mask.astype(np.float32) / info.max * alpha

    # LERP
    imageA = cv2.multiply(1. - mask, imageA)
    imageB = cv2.multiply(mask, imageB)
    imageA = cv2.add(imageA, imageB)
    return imageA.astype(np.uint8)

def comp_fill(image: TYPE_IMAGE,
              width: int,
              height: int,
              color: TYPE_PIXEL=1) -> TYPE_IMAGE:
    """
    Fills a block of pixels with a matte or stretched to width x height.
    """

    cc, chan = channel_count(image)
    y, x = image.shape[:2]
    canvas = channel_solid(width, height, color, chan=chan)
    y1 = max(0, (height - y) // 2)
    y2 = min(height, y1 + y)
    x1 = max(0, (width - x) // 2)
    x2 = min(width, x1 + x)
    canvas[y1: y2, x1: x2, :cc] = image
    return canvas

def comp_blend(imageA:Optional[TYPE_IMAGE]=None,
               imageB:Optional[TYPE_IMAGE]=None,
               mask:Optional[TYPE_IMAGE]=None,
               blendOp:BlendType=BlendType.NORMAL,
               alpha:float=1.,
               color:TYPE_PIXEL=0.,
               mode:EnumScaleMode=EnumScaleMode.NONE,
               resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:

    targetW, targetH = 0, 0
    if mode == EnumScaleMode.NONE:
        targetW = max(
            imageA.shape[1] if imageA is not None else 0,
            imageB.shape[1] if imageB is not None else 0,
            mask.shape[1] if mask is not None else 0
        )

        targetH = max(
            imageA.shape[0] if imageA is not None else 0,
            imageB.shape[0] if imageB is not None else 0,
            mask.shape[0] if mask is not None else 0
        )

    elif imageA is not None:
        targetH, targetW = imageA.shape[:2]
    elif imageB is not None:
        targetH, targetW = imageB.shape[:2]

    targetW, targetH = max(0, targetW), max(0, targetH)
    Logger.debug(targetW, targetH, imageA.shape if imageA else None, imageB.shape if imageB else None, blendOp, alpha, mode, resample)

    if targetH == 0 or targetW == 0:
        return channel_solid(targetW or 1, targetH or 1, )

    center = (targetW // 2, targetH // 2)

    def resize_input(img: TYPE_IMAGE) -> TYPE_IMAGE:
        if img is None:
            return channel_solid(targetW, targetH, 0., )

        h, w = img.shape[:2]
        if h != targetH or w != targetW:
            if mode != EnumScaleMode.NONE:
                img = geo_scalefit(img, targetW, targetH, mode, resample)
            img = comp_fill(img, targetW, targetH, color)
        return img

    imageA = resize_input(imageA)
    while channel_count(imageA)[0] < 3:
        imageA = channel_add(imageA, 0)
    if channel_count(imageA)[0] < 4:
        imageA = channel_add(imageA, 255)

    imageB = resize_input(imageB)
    while (cc := channel_count(imageB)[0]) < 3:
        imageB = channel_add(imageB, 0)

    if cc < 4:
        imageB = channel_add(imageB, 0)
        if mask is None:
            mask = channel_solid(targetW, targetH)
        elif mask is not None:
            if channel_count(mask)[0] != 1:
                # crunch the mask into an actual mask if RGB(A)
                mask = channel_solid(targetW, targetH, color_average(mask))

            hM, wM = mask.shape[:2]
            if hM != targetH or wM != targetW:
                if mode != EnumScaleMode.NONE:
                    mask = geo_scalefit(mask, targetW, targetH, mode, resample)
                mask = comp_fill(mask, targetW, targetH, 0)

    if cc < 4 or mask is not None:
        imageB[:, :, 3] = mask[:, :, 0]

    # make an image canvas to hold A and B that fits both on center
    if mode == EnumScaleMode.NONE:
        def xyz(img: TYPE_IMAGE) -> TYPE_IMAGE:
            h, w = img.shape[:2]
            y, x = max(0, center[1] - h // 2), max(0, center[0] - w // 2)
            canvas = channel_solid(targetW, targetH, 0, chan=EnumImageType.RGBA)
            canvas[y: y + h, x: x + w, :4] = img
            return canvas

        imageA = xyz(imageA)
        imageB = xyz(imageB)

    Logger.spam('end', imageA.shape, imageB.shape, mask.shape if mask is not None else 'NONE')

    imageA = cv2pil(imageA)
    imageB = cv2pil(imageB)
    image = blendLayers(imageA, imageB, blendOp, np.clip(alpha, 0, 1))
    return pil2cv(image)

# =============================================================================
# === ADJUST FUNCTIONS ===
# =============================================================================

def adjust_threshold(image:TYPE_IMAGE,
                     threshold:float=0.5,
                     mode:EnumThreshold=EnumThreshold.BINARY,
                     adapt:EnumThresholdAdapt=EnumThresholdAdapt.ADAPT_NONE,
                     block:int=3,
                     const:float=0.) -> TYPE_IMAGE:

    const = max(-100, min(100, const))
    block = max(3, block if block % 2 == 1 else block + 1)
    if adapt != EnumThresholdAdapt.ADAPT_NONE.value:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, adapt, cv2.THRESH_BINARY, block, const)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # gray = np.stack([gray, gray, gray], axis=-1)
        image = cv2.bitwise_and(image, gray)
    else:
        threshold = int(threshold * 255)
        _, image = cv2.threshold(image, threshold, 255, mode)
    return image

def adjust_levels(image: torch.Tensor, black_point:int=0, white_point=255,
                  mid_point=0.5, gamma=1.0) -> torch.Tensor:
    """
    Perform levels adjustment on a torch.tensor representing an image.

    Parameters:
    - image_tensor (torch.Tensor): Input image tensor.
    - black_point (float): Black point adjustment (default: 0).
    - white_point (float): White point adjustment (default: 255).
    - mid_point (float): Mid-point adjustment (default: 0.5).
    - gamma (float): Gamma adjustment (default: 1.0).

    Returns:
    - torch.Tensor: Adjusted image tensor.
    """
    # Apply black point adjustment
    image = torch.maximum(image - black_point, torch.tensor(0.0))

    # Apply white point adjustment
    image = torch.minimum(image, (white_point - black_point))

    # Apply mid-point adjustment
    image = (image + mid_point) - 0.5

    # Apply gamma adjustment
    image = torch.sign(image) * torch.pow(torch.abs(image), 1.0 / gamma)

    # Scale back to the range [0, 1]
    return (image + 0.5) / white_point

def adjust_sharpen(image: TYPE_IMAGE, kernel_size=None, sigma:float=1.0,
                   amount:float=1.0, threshold:float=0) -> TYPE_IMAGE:
    """Return a sharpened version of the image, using an unsharp mask."""

    kernel_size = (kernel_size, kernel_size) if kernel_size else (5, 5)
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def adjust_quantize(image: TYPE_IMAGE, levels:int=256, iterations:int=10, epsilon:float=0.2) -> TYPE_IMAGE:
    levels = int(max(2, min(256, levels)))
    pixels = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    _, labels, centers = cv2.kmeans(pixels, levels, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(image.shape)

def adjust_posterize(image: TYPE_IMAGE, levels:int=256) -> TYPE_IMAGE:
    divisor = 256 / max(2, min(256, levels))
    return (np.floor(image / divisor) * int(divisor)).astype(np.uint8)

def adjust_pixelate(image: TYPE_IMAGE, amount:float=1.)-> TYPE_IMAGE:

    h, w = image.shape[:2]
    amount = max(0, min(1, amount))
    block_size_h = max(1, (h * amount))
    block_size_w = max(1, (w * amount))
    num_blocks_h = int(np.ceil(h / block_size_h))
    num_blocks_w = int(np.ceil(w / block_size_w))
    block_size_h = h // num_blocks_h
    block_size_w = w // num_blocks_w
    pixelated_image = image.copy()

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Calculate block boundaries
            y_start = i * block_size_h
            y_end = min((i + 1) * block_size_h, h)
            x_start = j * block_size_w
            x_end = min((j + 1) * block_size_w, w)

            # Average color values within the block
            block_average = np.mean(image[y_start:y_end, x_start:x_end], axis=(0, 1))

            # Fill the block with the average color
            pixelated_image[y_start:y_end, x_start:x_end] = block_average

    return pixelated_image.astype(np.uint8)

# MORPHOLOGY

def morph_edge_detect(image: TYPE_IMAGE,
                      ksize: int=3,
                      low: float=0.27,
                      high:float=0.6) -> TYPE_IMAGE:

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ksize = max(3, ksize)
    image = cv2.GaussianBlur(src=image, ksize=(ksize, ksize+2), sigmaX=0.5)
    # Perform Canny edge detection
    return cv2.Canny(image, int(low * 255), int(high * 255))

def morph_emboss(image: TYPE_IMAGE, amount: float=1., kernel: int=2) -> TYPE_IMAGE:
    kernel = max(2, kernel)
    kernel = np.array([
        [-kernel,    -kernel+1,     0],
        [-kernel+1,    kernel-1,      1],
        [kernel-2,     kernel-1,      2]
    ]) * amount
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# KERNELS

def MEDIAN3x3(image: TYPE_IMAGE) -> TYPE_IMAGE:
    height, width, _ = image.shape
    out = np.zeros([height, width])
    for i in range(1, height-1):
        for j in range(1, width-1):
            temp = [
                image[i-1, j-1],
                image[i-1, j],
                image[i-1, j + 1],
                image[i, j-1],
                image[i, j],
                image[i, j + 1],
                image[i + 1, j-1],
                image[i + 1, j],
                image[i + 1, j + 1]
            ]

            temp = sorted(temp)
            out[i, j]= temp[4]
    return out

def kernel(stride: int) -> TYPE_IMAGE:
    """
    Generate a kernel matrix with a specific stride.

    The kernel matrix has a size of (stride, stride) and is filled with values
    such that if i < j, the element is set to -1; if i > j, the element is set to 1.

    Parameters:
    - stride (int): The size of the square kernel matrix.

    Returns:
    - TYPE_IMAGE: The generated kernel matrix.

    Example:
    >>> KERNEL(3)
    array([[ 0,  1,  1],
           [-1,  0,  1],
           [-1, -1,  0]], dtype=int8)
    """
    # Create an initial matrix of zeros
    kernel = np.zeros((stride, stride), dtype=np.int8)

    # Create a mask for elements where i < j and set them to -1
    mask_lower = np.tril(np.ones((stride, stride), dtype=bool), k=-1)
    kernel[mask_lower] = -1

    # Create a mask for elements where i > j and set them to 1
    mask_upper = np.triu(np.ones((stride, stride), dtype=bool), k=1)
    kernel[mask_upper] = 1

    return kernel

# =============================================================================
# === REMAPPING ===
# =============================================================================

def coord_sphere(width: int, height: int, radius: float) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    theta, phi = np.meshgrid(np.linspace(0, TAU, width), np.linspace(0, np.pi, height))

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    # z = radius * np.cos(phi)

    x_image = (x + 1) * (width - 1) / 2
    y_image = (y + 1) * (height - 1) / 2

    return x_image.astype(np.float32), y_image.astype(np.float32)

def coord_polar(width: int, height: int) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    rho = np.sqrt((map_x - width / 2)**2 + (map_y - height / 2)**2)
    phi = np.arctan2(map_y - height / 2, map_x - width / 2)
    return rho.astype(np.float32), phi.astype(np.float32)

def coord_perspective(width: int, height: int, pts: list[TYPE_COORD]) -> TYPE_IMAGE:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0] * width, pts[:, 1] * height])
    return cv2.getPerspectiveTransform(object_pts, pts)

def coord_fisheye(width: int, height: int, distortion: float) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    map_x, map_y = np.meshgrid(np.linspace(0., 1., width), np.linspace(0., 1., height))

    # normalized
    xnd, ynd = (2 * map_x - 1), (2 * map_y - 1)
    rd = np.sqrt(xnd**2 + ynd**2)

    # fish-eye distortion
    condition = (dist := 1 - distortion * (rd**2)) == 0
    xdu, ydu = np.where(condition, xnd, xnd / dist), np.where(condition, ynd, ynd / dist)
    xu, yu = ((xdu + 1) * width) / 2, ((ydu + 1) * height) / 2
    return xu.astype(np.float32), yu.astype(np.float32)

def remap_sphere(image: TYPE_IMAGE, radius: float) -> TYPE_IMAGE:
    height, width, _ = image.shape
    map_x, map_y = coord_sphere(width, height, radius)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_polar(image: TYPE_IMAGE) -> TYPE_IMAGE:
    height, width, _ = image.shape
    map_x, map_y = coord_polar(width, height)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_perspective(image: TYPE_IMAGE, pts: list) -> TYPE_IMAGE:
    height, width, _ = image.shape
    matrix: TYPE_IMAGE = coord_perspective(width, height, pts)
    return cv2.warpPerspective(image, matrix, (width, height))

def remap_fisheye(image: TYPE_IMAGE, distort: float) -> TYPE_IMAGE:
    height, width, _ = image.shape
    map_x, map_y = coord_fisheye(width, height, distort)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# =============================================================================
# === ZE MAIN ===
# =============================================================================

def testTRS() -> None:
    image = cv2.imread("./_res/img/alpha.png")

    pts = [
        [0.1, 0.1],
        [0.7, 0.3],
        [0.9, 0.9],
        [0.1, 0.9]
    ]
    remap = [
        ('perspective', remap_perspective(image, pts)),
        ('fisheye', remap_fisheye(image, 2)),
        ('sphere', remap_sphere(image, 0.1)),
        ('sphere', remap_sphere(image, 0.5)),
        ('sphere', remap_sphere(image, 1)),
        ('sphere', remap_sphere(image, 2)),
        ('polar', remap_polar(image)),
    ]
    idx_remap = 0
    while True:
        title, image,  = remap[idx_remap]
        cv2.imshow("", image)
        print(title)
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        idx_remap = (idx_remap + 1) % len(remap)

    cv2.destroyAllWindows()

def testColorConvert() -> None:
    Logger.debug(color_eval(1., EnumImageType.RGBA))
    Logger.debug(color_eval((1., 1.), EnumImageType.RGBA))
    Logger.debug(color_eval((1., 1., 1., 1.), EnumImageType.GRAYSCALE))
    Logger.debug(color_eval((255, 128, 100), EnumImageType.GRAYSCALE))
    Logger.debug(color_eval((255, 128, 100), EnumImageType.GRAYSCALE))
    Logger.debug(color_eval(255))

def testBlendModes() -> None:
    # all sizes and scale modes should work
    fore = cv2.imread('./_res/img/test_rainbow.png', cv2.IMREAD_UNCHANGED)
    back = cv2.imread('./_res/img/test_fore.png', cv2.IMREAD_UNCHANGED)
    # mask = cv2.imread('./_res/img/test_mask.png', cv2.IMREAD_UNCHANGED)
    mask = None
    for op in BlendType:
        for m in EnumScaleMode:
            a = comp_blend(None, fore, mask, blendOp=op, alpha=1., color=(255, 0, 0), mode=m)
            cv2.imwrite(f'./_res/tst/blend-{op.name}-{m.name}-0.png', a)
            a = comp_blend(back, None, mask, blendOp=op, alpha=0.5, color=(0, 255, 0), mode=m)
            cv2.imwrite(f'./_res/tst/blend-{op.name}-{m.name}-1.png', a)
            a = comp_blend(back, fore, None, blendOp=op, alpha=0, color=(0, 0, 255), mode=m)
            cv2.imwrite(f'./_res/tst/blend-{op.name}-{m.name}-2.png', a)

if __name__ == "__main__":
    testBlendModes()
