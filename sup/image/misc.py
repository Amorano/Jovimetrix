"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Extras Support
"""

import math
from enum import Enum
from typing import Any, Tuple

import cv2
import numpy as np
from numba import jit
from PIL import Image, ImageDraw

from loguru import logger

from Jovimetrix.sup.image import TYPE_IMAGE, TYPE_PIXEL, \
    TYPE_iRGB, bgr2image, image2bgr, image_convert, pil2cv

# =============================================================================
# === ENUMERATION ===
# =============================================================================

class EnumProjection(Enum):
    NORMAL = 0
    POLAR = 5
    SPHERICAL = 10
    FISHEYE = 15
    PERSPECTIVE = 20

class EnumShapes(Enum):
    CIRCLE = 0
    SQUARE = 1
    ELLIPSE = 2
    RECTANGLE = 3
    POLYGON = 4

class EnumThreshold(Enum):
    BINARY = cv2.THRESH_BINARY
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO

class EnumThresholdAdapt(Enum):
    ADAPT_NONE = -1
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADAPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# =============================================================================
# === EXPLICIT SHAPE FUNCTIONS ===
# =============================================================================

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

def image_gradient(width:int, height:int, color_map:dict=None) -> TYPE_IMAGE:
    if color_map is None:
        color_map = {0: (0,0,0,255)}
    else:
        color_map = {np.clip(float(k), 0, 1): [np.clip(int(c), 0, 255) for c in v] for k, v in color_map.items()}
    color_map = dict(sorted(color_map.items()))
    image = Image.new('RGBA', (width, height))
    draw = image.load()
    widthf = float(width)

    @jit
    def gaussian(x, a, b, c, d=0) -> Any:
        return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

    def pixel(x, spread:int=1) -> TYPE_iRGB:
        ws = widthf / (spread * len(color_map))
        r = sum([gaussian(x, p[0], k * widthf, ws) for k, p in color_map.items()])
        g = sum([gaussian(x, p[1], k * widthf, ws) for k, p in color_map.items()])
        b = sum([gaussian(x, p[2], k * widthf, ws) for k, p in color_map.items()])
        return min(255, int(r)), min(255, int(g)), min(255, int(b))

    for x in range(width):
        r, g, b = pixel(x)
        for y in range(height):
            draw[x, y] = r, g, b
    return pil2cv(image)

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

def image_stereogram(image: TYPE_IMAGE, depth: TYPE_IMAGE, divisions:int=8,
                     mix:float=0.33, gamma:float=0.33, shift:float=1.) -> TYPE_IMAGE:
    height, width = depth.shape[:2]
    out = np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.resize(image, (width, height))
    image = image_convert(image, 3)
    depth = image_convert(depth, 3)
    noise = np.random.randint(0, max(1, int(gamma * 255)), (height, width, 3), dtype=np.uint8)
    # noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    image = cv2.addWeighted(image, 1. - mix, noise, mix, 0)

    pattern_width = width // divisions
    # shift -= 1
    for y in range(height):
        for x in range(width):
            if x < pattern_width:
                out[y, x] = image[y, x]
            else:
                # out[y, x] = out[y, x - pattern_width + int(shift * invert)]
                offset = depth[y, x][0] // divisions
                pos = x - pattern_width + int(shift * offset)
                # pos = max(-pattern_width, min(pattern_width, pos))
                out[y, x] = out[y, pos]
    return out

def image_threshold(image:TYPE_IMAGE, threshold:float=0.5,
                    mode:EnumThreshold=EnumThreshold.BINARY,
                    adapt:EnumThresholdAdapt=EnumThresholdAdapt.ADAPT_NONE,
                    block:int=3, const:float=0.) -> TYPE_IMAGE:

    const = max(-100, min(100, const))
    block = max(3, block if block % 2 == 1 else block + 1)
    image, alpha, cc = image2bgr(image)
    if adapt != EnumThresholdAdapt.ADAPT_NONE:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, adapt.value, cv2.THRESH_BINARY, block, const)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # gray = np.stack([gray, gray, gray], axis=-1)
        image = cv2.bitwise_and(image, gray)
    else:
        threshold = int(threshold * 255)
        _, image = cv2.threshold(image, threshold, 255, mode.value)
    return bgr2image(image, alpha, cc == 1)

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
        [-kernel,   -kernel+1,    0],
        [-kernel+1,   kernel-1,     1],
        [kernel-2,    kernel-1,     2]
    ]) * amount
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
