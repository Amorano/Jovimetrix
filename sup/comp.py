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

import cv2
import torch
import numpy as np
from scipy.ndimage import rotate
from PIL import Image, ImageDraw, ImageChops

from enum import Enum
from typing import Any

try:
    from .util import loginfo, logwarn, logerr
except:
    from sup.util import loginfo, logwarn, logerr

# =============================================================================
# === GLOBAL ENUMS ===
# =============================================================================

class EnumThreshold(Enum):
    BINARY = cv2.THRESH_BINARY
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO
EnumThresholdName = [e.name for e in EnumThreshold]

class EnumAdaptThreshold(Enum):
    ADAPT_NONE = -1
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADAPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
EnumAdaptThresholdName = [e.name for e in EnumAdaptThreshold]

class EnumOPBlend(Enum):
    LERP = None
    ADD = ImageChops.add
    MINIMUM = ImageChops.darker
    MAXIMUM = ImageChops.lighter
    MULTIPLY = ImageChops.multiply
    SOFT_LIGHT = ImageChops.soft_light
    HARD_LIGHT = ImageChops.hard_light
    OVERLAY = ImageChops.overlay
    SCREEN = ImageChops.screen
    SUBTRACT = ImageChops.subtract
    DIFFERENCE = ImageChops.difference
    LOGICAL_AND = np.bitwise_and
    LOGICAL_OR = np.bitwise_or
    LOGICAL_XOR = np.bitwise_xor
EnumOPBlendName = [e.name for e in EnumThreshold]

OP_BLEND = {
    'LERP': "",
    'ADD': ImageChops.add,
    'MINIMUM': ImageChops.darker,
    'MAXIMUM': ImageChops.lighter,
    'MULTIPLY': ImageChops.multiply,
    'SOFT LIGHT': ImageChops.soft_light,
    'HARD LIGHT': ImageChops.hard_light,
    'OVERLAY': ImageChops.overlay,
    'SCREEN': ImageChops.screen,
    'SUBTRACT': ImageChops.subtract,
    'DIFFERENCE': ImageChops.difference,
    'LOGICAL AND': np.bitwise_and,
    'LOGICAL OR': np.bitwise_or,
    'LOGICAL XOR': np.bitwise_xor,
}

# =============================================================================
# === MATRIX SUPPORT ===
# =============================================================================

def tensor2pil(tensor: torch.Tensor) -> Image:
    """Torch Tensor to PIL Image."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

def tensor2cv(tensor: torch.Tensor) -> cv2.Mat:
    """Torch Tensor to CV2 Matrix."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)

def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    """Torch Tensor to Numpy Array."""
    return np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

def mask2pil(mask: torch.Tensor) -> Image:
    """Torch Tensor (Mask) to PIL."""
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask = np.clip(255 * mask.cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(mask, mode='L')

def pil2tensor(image: Image) -> torch.Tensor:
    """PIL Image to Torch Tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2cv(image: Image) -> cv2.Mat:
    """PIL to CV2 Matrix."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def pil2mask(image: Image) -> torch.Tensor:
    """PIL Image to Torch Tensor (Mask)."""
    image = np.array(image.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(image)

def cv2pil(image: cv2.Mat) -> Image:
    """CV2 Matrix to PIL."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

def cv2tensor(image: cv2.Mat) -> torch.Tensor:
    """CV2 Matrix to Torch Tensor."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(image / 255.0).unsqueeze(0)

def cv2mask(image: cv2.Mat) -> torch.Tensor:
    """CV2 to Torch Tensor (Mask)."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return torch.from_numpy(image / 255.0).unsqueeze(0)

# =============================================================================
# === SHAPE FUNCTIONS ===
# =============================================================================

def sh_body(func: str, width: int, height: int, sizeX=1., sizeY=1., fill=(255, 255, 255)) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    func = getattr(d, func)
    func(xy, fill=fill)
    return image

def sh_ellipse(width: int, height: int, sizeX=1., sizeY=1., fill=None) -> Image:
    return sh_body('ellipse', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

def sh_quad(width: int, height: int, sizeX=1., sizeY=1., fill=None) -> Image:
    return sh_body('rectangle', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

def sh_polygon(width: int, height: int, size: float=1., sides: int=3, angle: float=0., fill=None) -> Image:
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

def CROP(image: cv2.Mat, x1: int, y1: int, x2: int, y2: int) -> cv2.Mat:
    """."""
    height, width, _ = image.shape

    x1 = min(max(0, x1), width)
    x2 = min(max(0, x2), width)
    y1 = min(max(0, y1), height)
    y2 = min(max(0, y2), height)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    loginfo(f"CROP ({x1}, {y1}) :: ({x2}, {y2}) [{width}x{height}]")
    return image[y1:y2, x1:x2]

def CROP_CENTER(image: cv2.Mat, targetW: int, targetH: int) -> cv2.Mat:
    """AUTO Center CROP based on image and target size."""
    height, width, _ = image.shape
    h_center = int(height * 0.5)
    w_center = int(width * 0.5)
    w_delta = int(targetW * 0.5)
    h_delta = int(targetH * 0.5)
    loginfo(f"CROP_CENTER [{w_center}, {h_center}]  [{w_delta}, {h_delta}]")
    return CROP(image, w_center - w_delta, h_center - h_delta, w_center + w_delta, h_center + h_delta)

def EDGEWRAP(image: cv2.Mat, tileX: float=1., tileY: float=1., edge: str='WRAP') -> cv2.Mat:
    """TILING."""
    height, width, _ = image.shape
    tileX = int(tileX * width * 1) if edge in ["WRAP", "WRAPX"] else 0
    tileY = int(tileY * height * 1) if edge in ["WRAP", "WRAPY"] else 0
    loginfo(f"EDGEWRAP [{width}, {height}]  [{tileX}, {tileY}]")
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

def TRANSLATE(image: cv2.Mat, offsetX: float, offsetY: float) -> cv2.Mat:
    """TRANSLATION."""
    height, width, _ = image.shape
    M = np.float32([[1, 0, offsetX * width], [0, 1, offsetY * height]])
    loginfo(f"TRANSLATE [{offsetX}, {offsetY}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def ROTATE(image: cv2.Mat, angle: float, center=(0.5 ,0.5)) -> cv2.Mat:
    """ROTATION."""
    height, width, _ = image.shape
    center = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    loginfo(f"ROTATE [{angle}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def ROTATE_NDARRAY(image: np.ndarray, angle: float, clip: bool=True) -> np.ndarray:
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

def SCALEFIT(image: cv2.Mat, width: int, height: int, mode: str) -> cv2.Mat:
    """Scale a matrix into a defined width, height explicitly or by a guiding edge."""
    if mode == "ASPECT":
        h, w, _ = image.shape
        scalar = max(width, height)
        scalar /= max(w, h)
        return cv2.resize(image, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_AREA)
    elif mode == "CROP":
        return CROP_CENTER(image, width, height)
    elif mode == "FIT":
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image

def TRANSFORM(image: cv2.Mat, offsetX: float=0., offsetY: float=0., angle: float=0., sizeX: float=1., sizeY: float=1., edge:str='CLIP', widthT: int=256, heightT: int=256, mode: str='FIX') -> cv2.Mat:
    """Transform, Rotate and Scale followed by Tiling and then Inversion, conforming to an input wT, hT,."""
    height, width, _ = image.shape

    # SCALE
    if (sizeX != 1. or sizeY != 1.) and edge != "CLIP":
        tx = ty = 0
        if edge in ["WRAP", "WRAPX"] and sizeX < 1.:
            tx = 1. / sizeX - 1
            sizeX = 1.

        if edge in ["WRAP", "WRAPY"] and sizeY < 1.:
            ty = 1. / sizeY - 1
            sizeY = 1.
        image = EDGEWRAP(image, tx, ty)
        h, w, _ = image.shape
        loginfo(f"EDGEWRAP_POST [{w}, {h}]")

    if sizeX != 1. or sizeY != 1.:
        wx = int(width * sizeX)
        hx = int(height * sizeY)
        loginfo(f"SCALE [{wx}, {hx}]")
        image = cv2.resize(image, (wx, hx), interpolation=cv2.INTER_AREA)

    #if edge != "CLIP":
    #    image = CROP_CENTER(image, width, height)

    # ROTATION
    if angle != 0:
        if edge != "CLIP":
            image = EDGEWRAP(image)
        image = ROTATE(image, angle)

    # TRANSLATION
    if offsetX != 0. or offsetY != 0.:
        if edge != "CLIP":
            image = EDGEWRAP(image)
        image = TRANSLATE(image, offsetX, offsetY)
        if edge != "CLIP":
            image = CROP_CENTER(image, width, height)

    return SCALEFIT(image, widthT, heightT, mode=mode)

def HSV(image: cv2.Mat, hue: float, saturation: float, value: float) -> cv2.Mat:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue *= 255
    image[:, :, 0] = (image[:, :, 0] + hue) % 180
    image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
    loginfo(f"HSV {hue} {saturation} {value}")
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def GAMMA(image: cv2.Mat, value: float) -> cv2.Mat:
    image = np.clip(cv2.pow(image / 255, value), 0, 1)
    loginfo(f"GAMMA {image.dtype} ({value})")
    return np.uint8(image * 255)

def CONTRAST(image: cv2.Mat, value: float) -> cv2.Mat:
    image = np.clip((image / 255 - 0.5) * value + 0.5, 0, 1)
    loginfo(f"CONTRAST {image.dtype} ({value})")
    return np.uint8(image * 255)

def EXPOSURE(image: cv2.Mat, value: float) -> cv2.Mat:
    image = np.clip(image / 255 * (2.0 ** value), 0, 1)
    loginfo(f"EXPOSURE {image.dtype} ({value})")
    return np.uint8(image * 255)

def INVERT(image: cv2.Mat, value: float) -> cv2.Mat:
    value = np.clip(value, 0, 1)
    inverted = np.abs(255 - image)
    image = cv2.addWeighted(image, 1 - value, inverted, value, 0)
    return image

def MIRROR(image: cv2.Mat, pX: float, axis: int, invert: bool=False) -> cv2.Mat:
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

def EXTEND(imageA: cv2.Mat, imageB: cv2.Mat, axis: int=0, flip: bool=False) -> cv2.Mat:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def LERP(imageA: cv2.Mat, imageB: cv2.Mat, mask: cv2.Mat=None, alpha: float=1.) -> cv2.Mat:
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

def BLEND(imageA: cv2.Mat, imageB: cv2.Mat, func: str, width: int, height: int,
          mask: cv2.Mat=None, alpha: float=1.) -> cv2.Mat:

    if (op := OP_BLEND.get(func, None)) is None:
        return imageA

    alpha = np.clip(alpha, 0, 1)
    if mask is None:
        height, width, _ = imageA.shape
        mask = cv2.empty((height, width, 1), dtype=cv2.uint8)

    # recale images to match sourceA size...
    def adjustSize(who: cv2.Mat) -> cv2.Mat:
        h, w, _ = who.shape
        if (w != width or h != height):
            return SCALEFIT(who, width, height)
        return who

    imageA = adjustSize(imageA)
    imageB = adjustSize(imageB)
    mask = adjustSize(mask)

    if func.startswith("LOGICAL"):
        imageB = op(imageA, imageB)
        # imageB = pil2cv(Image.fromarray(imageB))
    elif func != "LERP":
        imageB = pil2cv(op(cv2pil(imageA), cv2pil(imageB)))

    # take the new B and mix with mask and alpha
    return LERP(imageA, imageB, mask, alpha)

def THRESHOLD(image: cv2.Mat, threshold: float=0.5, mode: EnumThreshold=EnumThreshold.BINARY,
              adapt: EnumAdaptThreshold=EnumAdaptThreshold.ADAPT_NONE, block: int=3, const: float=0.) -> cv2.Mat:

    if adapt != EnumAdaptThreshold.ADAPT_NONE.value:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(gray, 255, adapt, cv2.THRESH_BINARY, block, const)
        image = cv2.multiply(gray, image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        threshold = int(threshold * 255)
        _, image = cv2.threshold(image, threshold, 255, mode)
    return image
