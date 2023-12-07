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

from Jovimetrix import Logger, grid_make, cv2mask, cv2tensor, cv2pil, pil2cv

HALFPI = math.pi / 2
TAU = math.pi * 2

# =============================================================================
# === ENUM GLOBALS ===
# =============================================================================

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
    SHARPEN = 4
    EMBOSS = 5
    FIND_EDGES = 6
    OUTLINE = 7
    DILATE = 8
    ERODE = 9
    OPEN = 10
    CLOSE = 11

EnumBlendType = [str(x).split('.')[-1] for x in BlendType]

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

def gray_sized(image: cv2.Mat, h:int, w:int, resample: EnumInterpolation=EnumInterpolation.LANCZOS4) -> cv2.Mat:
    """Force an image into Grayscale at a specific width, height."""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[0] != h or image.shape[1] != w:
        image = cv2.resize(image, (w, h), interpolation=resample.value)
    return image

def pixel_split(image: cv2.Mat) -> tuple[list[cv2.Mat], list[cv2.Mat]]:
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

def merge_channel(channel, size, resample: EnumInterpolation=EnumInterpolation.LANCZOS4) -> cv2.Mat:
    if channel is None:
        return np.full(size, 0, dtype=np.uint8)
    return gray_sized(channel, *size[::-1], resample)

def pixel_merge(r: cv2.Mat, g: cv2.Mat, b: cv2.Mat, a: cv2.Mat,
          width: int, height: int,
          mode:EnumScaleMode=EnumScaleMode.NONE,
          resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> cv2.Mat:

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

def shape_body(func: str, width: int, height: int, sizeX=1., sizeY=1., fill=(255, 255, 255)) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    func = getattr(d, func)
    func(xy, fill=fill)
    return image

def shape_ellipse(width: int, height: int, sizeX=1., sizeY=1., fill=None) -> Image:
    return shape_body('ellipse', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

def shape_quad(width: int, height: int, sizeX=1., sizeY=1., fill=None) -> Image:
    return shape_body('rectangle', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

def shape_polygon(width: int, height: int, size: float=1., sides: int=3, angle: float=0., fill=None) -> Image:
    fill=fill or (255, 255, 255)
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image

# =============================================================================
# === IMAGE FUNCTIONS ===
# =============================================================================

# IMAGE

def image_stack(images: list[np.ndarray[np.uint8]],
                axis:Optional[EnumOrientation]=EnumOrientation.HORIZONTAL,
                stride:Optional[int]=None,
                color:Optional[tuple[float, float, float]]=(0,0,0),
                mode:EnumScaleMode=EnumScaleMode.NONE,
                resample:Optional[Image.Resampling]=Image.Resampling.LANCZOS) -> np.ndarray[np.uint8]:

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

    center = (width // 2, height // 2)
    images = [comp_fill(i, center, width, height, color, mode, resample) for i in images]

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

def image_grid(data: list[np.ndarray[np.uint8]], width: int, height: int) -> np.ndarray:
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

# GEOMETRY

def geo_crop(image: cv2.Mat, left=None, top=None, right=None, bottom=None,
             widthT: int=None, heightT: int=None, pad:bool=False,
             color: tuple[float, float, float]=(0, 0, 0)) -> cv2.Mat:

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

        img_padded = np.full((heightT, widthT, 3), color, dtype=np.uint8)

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

def geo_edge_wrap(image: np.ndarray[np.uint8], tileX: float=1., tileY: float=1., edge: str='WRAP') -> np.ndarray[np.uint8]:
    """TILING."""
    height, width, _ = image.shape
    tileX = int(tileX * width * 0.5) if edge in ["WRAP", "WRAPX"] else 0
    tileY = int(tileY * height * 0.5) if edge in ["WRAP", "WRAPY"] else 0
    # Logger.debug("geo_edge_wrap", f"[{width}, {height}]  [{tileX}, {tileY}]")
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

def geo_translate(image: np.ndarray[np.uint8], offsetX: float, offsetY: float) -> np.ndarray[np.uint8]:
    """TRANSLATION."""
    height, width, _ = image.shape
    M = np.float32([[1, 0, offsetX * width], [0, 1, offsetY * height]])
    # Logger.debug("geo_translate", f"[{offsetX}, {offsetY}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def geo_rotate(image: np.ndarray[np.uint8], angle: float, center=(0.5 ,0.5)) -> np.ndarray[np.uint8]:
    """ROTATION."""
    height, width, _ = image.shape
    center = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # Logger.debug("geo_rotate", f"[{angle}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def geo_rotate_array(image: np.ndarray[np.uint8], angle: float, clip: bool=True) -> np.ndarray[np.uint8]:
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

def geo_scalefit(image: np.ndarray[np.uint8], width: int, height:int,
                 mode:EnumScaleMode=EnumScaleMode.NONE,
                 resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> np.ndarray[np.uint8]:

    # logspam("geo_scalefit", mode, f"[{width}x{height}]", f"rs=({resample})")

    match mode:
        case EnumScaleMode.ASPECT:
            h, w, _ = image.shape
            scalar = max(width, height)
            scalar /= max(w, h)
            return cv2.resize(image, None, fx=scalar, fy=scalar, interpolation=resample.value)

        case EnumScaleMode.CROP:
            return geo_crop(image, widthT=width, heightT=height, pad=True)

        case EnumScaleMode.FIT:
            return cv2.resize(image, (width, height), interpolation=resample.value)

    return image

def geo_transform(image: np.ndarray[np.uint8], offsetX: float=0., offsetY: float=0., angle: float=0.,
              sizeX: float=1., sizeY: float=1., edge:str='CLIP', widthT: int=256, heightT: int=256,
              mode:EnumScaleMode=EnumScaleMode.NONE,
              resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> np.ndarray[np.uint8]:
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

def geo_merge(imageA: np.ndarray[np.uint8], imageB: np.ndarray[np.uint8], axis: int=0, flip: bool=False) -> np.ndarray[np.uint8]:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def geo_mirror(image: np.ndarray[np.uint8], pX: float, axis: int, invert: bool=False) -> np.ndarray[np.uint8]:
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

# LIGHT / COLOR

def light_hsv(image: np.ndarray[np.uint8], hue: float, saturation: float, value: float) -> np.ndarray[np.uint8]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue *= 255
    image[:, :, 0] = (image[:, :, 0] + hue) % 180
    image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
    # Logger.debug("light_hsv", f"{hue} {saturation} {value}")
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def light_gamma(image: np.ndarray[np.uint8], value: float) -> np.ndarray[np.uint8]:
    # Logger.debug("light_gamma", f"({value})")
    if value == 0:
        return (image * 0).astype(np.uint8)

    invGamma = 1.0 / max(0.000001, value)
    lookUpTable = np.clip(np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype(np.uint8), 0, 255)
    return cv2.LUT(image, lookUpTable)

def light_contrast(image: np.ndarray[np.uint8], value: float) -> np.ndarray[np.uint8]:
    # Logger.debug("light_contrast", f"({value})")
    mean_value = np.mean(image)
    image = (image - mean_value) * value + mean_value
    return np.clip(image, 0, 255).astype(np.uint8)

def light_exposure(image: np.ndarray[np.uint8], value: float) -> np.ndarray[np.uint8]:
    # Logger.debug("light_exposure", f"({math.pow(2.0, value)})")
    return np.clip(image * value, 0, 255).astype(np.uint8)

def light_invert(image: np.ndarray[np.uint8], value: float) -> np.ndarray[np.uint8]:
    value = np.clip(value, 0, 255)
    inverted = np.abs(255 - image)
    return cv2.addWeighted(image, 1 - value, inverted, value, 0)

def color_match(image: np.ndarray[np.uint8],
                usermap: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
    """Colorize one input based on the histogram matches."""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    beta = cv2.cvtColor(usermap, cv2.COLOR_BGR2LAB)
    image = exposure.match_histograms(image, beta, channel_axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return comp_blend(usermap, image, blendOp=BlendType.LUMINOSITY)

def color_colormap(image: np.ndarray[np.uint8],
                   usermap: Optional[np.ndarray[np.uint8]]=None,
                   colormap: Optional[int]=cv2.COLORMAP_JET) -> np.ndarray[np.uint8]:
    """Colorize one input based on custom GNU Octave/MATLAB map"""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image[:, :, 1]
    if usermap is not None:
        return cv2.applyColorMap(image, usermap)
    return cv2.applyColorMap(image, colormap)

def color_heatmap(image: np.ndarray[np.uint8],
                  colormap:int=cv2.COLORMAP_JET,
                  threshold:float=0.55,
                  sigma:int=13) -> np.ndarray[np.uint8]:
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

# COMP

def channel_count(image) -> None:
    if (len(image.shape) == 2 or
        (len(image.shape) == 3 and
         image.shape[2] == 1)):
        return 1
    return image.shape[2]

def pixel_block(width:int, height:int,
                color:tuple[float, float, float]=(0,0,0),
                chan:int=1) -> np.ndarray[np.uint8]:

    if chan == 3:
        image = np.zeros((height, width, chan), dtype=np.uint8)
        color = tuple(int(max(0, min(1, c)) * 255) for c in color)
        image[:, :] = color
        return image

    image = np.zeros((height, width), dtype=np.uint8)
    luma = [int(max(0, min(1, c))) for c in color]
    image[:, :] = max(*luma) * 255
    return image

def comp_fill(image: np.ndarray[np.uint8],
              center: tuple[int, int],
              width:int, height:int,
              fill:tuple[float, float, float]=(1,1,1),
              mode:EnumScaleMode=EnumScaleMode.NONE,
              resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> np.ndarray[np.uint8]:
    """
    Fills a block of pixels with a matte or stretched to width x height.
    """
    cc = channel_count(image)
    if mode != EnumScaleMode.NONE:
        image = geo_scalefit(image, width, height, mode, resample)
        if cc != 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        if cc == 3:
            blank = pixel_block(width, height, fill, chan=3)
            image[:, :, :3] = blank
        return image

    y, x = image.shape[:2]
    blank = pixel_block(x, y, fill, chan=3)
    y1 = max(0, center[1] - y // 2)
    y2 = min(y, center[1] + (y + 1) // 2)
    x1 = max(0, center[0] - x // 2)
    x2 = min(x, center[0] + (x + 1) // 2)
    if cc == 1:
        blank[y1: y2, x1: x2] = image
    elif cc > 2:
        blank[y1: y2, x1: x2, :3] = image
    return blank

def comp_lerp(imageA:np.ndarray[np.uint8],
              imageB:np.ndarray[np.uint8],
              mask:np.ndarray=None,
              alpha:float=1.) -> np.ndarray[np.uint8]:

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

def comp_blend(imageA:Optional[np.ndarray[np.uint8]]=None,
               imageB:Optional[np.ndarray[np.uint8]]=None,
               mask:Optional[np.ndarray[np.uint8]]=None,
               blendOp:BlendType=BlendType.NORMAL,
               alpha:float=1,
               fill:tuple[float, float, float]=(1,1,1),
               mode:EnumScaleMode=EnumScaleMode.NONE,
               resample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> np.ndarray[np.uint8]:

    # Determine the maximum sizes among imageA, imageB
    max_width = max(
        imageA.shape[1] if imageA is not None else 0,
        imageB.shape[1] if imageB is not None else 0,
        mask.shape[1] if mask is not None else 0
    )

    max_height = max(
        imageA.shape[0] if imageA is not None else 0,
        imageB.shape[0] if imageB is not None else 0,
        mask.shape[0] if mask is not None else 0
    )

    Logger.debug('comp_blend', max_width, max_height, blendOp, alpha, mode, resample)
    Logger.debug('comp_blend', imageA.shape, imageB.shape, mask.shape)

    if max_width == 0 and max_height == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    center = (max_width // 2, max_height // 2)

    cc = channel_count(imageA)
    height, width = imageA.shape[:2]
    if cc != 4 or height != max_height or width != max_width:
        imageA = comp_fill(imageA, center, max_width, max_height, fill, mode, resample)

    cc = channel_count(imageB)
    height, width = imageB.shape[:2]
    if cc != 4 or height != max_height or width != max_width:
        imageB = comp_fill(imageB, center, max_width, max_height, fill, mode, resample)

    if mask is None:
        mask = np.ones((max_height, max_width), dtype=np.uint8) * 255
    else:
        height, width = mask.shape[:2]
        if height != max_height or width != max_width:
            mask = comp_fill(mask, center, max_width, max_height, fill, mode, resample)

    cc = channel_count(mask)
    if cc != 1:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    Logger.debug('comp_blend', imageA.shape, imageB.shape, mask.shape)
    imageB[:, :, 3] = mask

    # Logger.debug('comp_blend', imageA.shape, imageB.shape, mask.shape)

    imageA = cv2pil(imageA)
    imageB = cv2pil(imageB)
    image = blendLayers(imageA, imageB, blendOp, np.clip(alpha, 0, 1))
    return pil2cv(image)

# ADJUST

def adjust_threshold(image:np.ndarray[np.uint8],
                     threshold:float=0.5,
                     mode:EnumThreshold=EnumThreshold.BINARY,
                     adapt:EnumThresholdAdapt=EnumThresholdAdapt.ADAPT_NONE,
                     block:int=3,
                     const:float=0.) -> np.ndarray[np.uint8]:

    if adapt != EnumThresholdAdapt.ADAPT_NONE:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(gray, 255, adapt, cv2.THRESH_BINARY, block, const)
        image = cv2.multiply(gray, image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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

def adjust_sharpen(image: np.ndarray[np.uint8], kernel_size=None, sigma:float=1.0,
                   amount:float=1.0, threshold:float=0) -> np.ndarray[np.uint8]:
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

# MORPHOLOGY

def morph_edge_detect(image: np.ndarray[np.uint8], low: float=0.27,
                      high:float=0.6) -> np.ndarray[np.uint8]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(src=image, ksize=(3, 5), sigmaX=0.5)
    # Perform Canny edge detection
    return cv2.Canny(image, int(low * 255), int(high * 255))

def morph_emboss(image: np.ndarray[np.uint8], amount: float=1.) -> np.ndarray[np.uint8]:
    kernel = np.array([
        [-2,    -1,     0],
        [-1,    1,      1],
        [0,     1,      2]
    ]) * amount
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# KERNELS

def MEDIAN3x3(image: np.ndarray) -> np.ndarray[np.uint8]:
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

def kernel(stride: int) -> np.ndarray[np.uint8]:
    """
    Generate a kernel matrix with a specific stride.

    The kernel matrix has a size of (stride, stride) and is filled with values
    such that if i < j, the element is set to -1; if i > j, the element is set to 1.

    Parameters:
    - stride (int): The size of the square kernel matrix.

    Returns:
    - np.ndarray: The generated kernel matrix.

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

def coord_sphere(width: int, height: int, radius: float) -> tuple[np.ndarray[np.uint8], np.ndarray]:
    theta, phi = np.meshgrid(np.linspace(0, TAU, width), np.linspace(0, np.pi, height))

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    # z = radius * np.cos(phi)

    x_image = (x + 1) * (width - 1) / 2
    y_image = (y + 1) * (height - 1) / 2

    return x_image.astype(np.float32), y_image.astype(np.float32)

def coord_polar(width: int, height: int) -> tuple[np.ndarray[np.uint8], np.ndarray]:
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    rho = np.sqrt((map_x - width / 2)**2 + (map_y - height / 2)**2)
    phi = np.arctan2(map_y - height / 2, map_x - width / 2)
    return rho.astype(np.float32), phi.astype(np.float32)

def coord_perspective(width: int, height: int, pts: list[tuple[int, int]]) -> np.ndarray[np.uint8]:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0] * width, pts[:, 1] * height])
    return cv2.getPerspectiveTransform(object_pts, pts)

def coord_fisheye(width: int, height: int, distortion: float) -> tuple[np.ndarray[np.uint8], np.ndarray]:
    map_x, map_y = np.meshgrid(np.linspace(0., 1., width), np.linspace(0., 1., height))

    # normalized
    xnd, ynd = (2 * map_x - 1), (2 * map_y - 1)
    rd = np.sqrt(xnd**2 + ynd**2)

    # fish-eye distortion
    condition = (dist := 1 - distortion * (rd**2)) == 0
    xdu, ydu = np.where(condition, xnd, xnd / dist), np.where(condition, ynd, ynd / dist)
    xu, yu = ((xdu + 1) * width) / 2, ((ydu + 1) * height) / 2
    return xu.astype(np.float32), yu.astype(np.float32)

def remap_sphere(image: np.ndarray[np.uint8], radius: float) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    map_x, map_y = coord_sphere(width, height, radius)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_polar(image: np.ndarray) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    map_x, map_y = coord_polar(width, height)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_perspective(image: np.ndarray[np.uint8], pts: list) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    matrix: np.ndarray = coord_perspective(width, height, pts)
    return cv2.warpPerspective(image, matrix, (width, height))

def remap_fisheye(image: np.ndarray[np.uint8], distort: float) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    map_x, map_y = coord_fisheye(width, height, distort)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# =============================================================================
# === ZE MAIN ===
# =============================================================================

def testTRS():
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

if __name__ == "__main__":
    # 200 x 200
    back = cv2.imread('./_res/img/404.png', cv2.IMREAD_UNCHANGED)
    front = cv2.imread('./_res/img/alpha.png', cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('./_res/img/beta.png', cv2.IMREAD_UNCHANGED)
    a = comp_blend(back, front, mask, blendOp=BlendType.DIFFERENCE, alpha=1, mode=EnumScaleMode.NONE)
    cv2.imwrite('./_res/img/_ground.png', a)
