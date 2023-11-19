"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix
"""

import torch
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import rotate

# =============================================================================
# === BASE NODE FOR ALL ===
# =============================================================================
class JovimetrixBaseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{}}

    DESCRIPTION = "A Jovimetrix Node"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    OUTPUT_NODE = True
    FUNCTION = "run"

# =============================================================================
# === IMAGE SUPPORT ===
# =============================================================================
# Torch Tensor to PIL
def tensor2pil(image: torch.Tensor) -> Image:
        image = 255. * image.cpu().numpy().squeeze()
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return Image.fromarray(image)

# Torch Tensor to CV2
def tensor2cv(image: torch.Tensor) -> cv2.Mat:
    M = 255. * image.cpu().numpy().squeeze()
    image = Image.fromarray(np.clip(M, 0, 255).astype(np.uint8)) #.convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Torch Tensor to Numpy
def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    return np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

# PIL to Tensor RGB
def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL to NUMPY
def pil2np(image: Image) -> np.ndarray:
    return (np.array(image).astype(np.float32) / 255.0)[ :, :, :]

# PIL to CV2 Matrix
def pil2cv(image: Image) -> cv2.Mat:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# CV2 Matrix to PIL
def cv2pil(image: cv2.Mat) -> Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# CV2 to Torch Tensor
def cv2tensor(image: cv2.Mat) -> torch.Tensor:
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def cv2mask(image: cv2.Mat) -> torch.Tensor:
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL to Tensor RGB
def np2tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

# re-range -0.5...+0.5 to 0..1 scaled and clipped to scalar
def range2one(x: float) -> float:
    return np.clip(x, -0.5, 0.5) + 0.5

def rotate_ndarray(image, angle, clip=True):
    rotated_image = rotate(image, angle, reshape=not clip, mode='constant', cval=0)

    if not clip:
        return rotated_image

    # Compute the dimensions for clipping
    # print(image.shape, rotated_image.shape)
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

# @TODO: MAKE THIS WORK FOR CROPPING GENERALLY
def CROP(image: cv2.Mat, x1: int, y1: int, x2: int, y2: int) -> cv2.Mat:
    height, width, _ = image.shape
    x1 = min(max(0, x1), width)
    x2 = min(max(0, x2), width)
    y1 = min(max(0, y1), height)
    y2 = min(max(0, y2), height)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    cropped = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
    image[y1:y2, x1:x2] = cropped
    return cropped

# AUTO Center CROP based on image and target size
def CROP_CENTER(image: cv2.Mat, targetW: int, targetH: int) -> cv2.Mat:
    height, width, _ = image.shape
    h_center = int(height * 0.5)
    w_center = int(width * 0.5)
    w_delta = int(targetW * 0.5)
    h_delta = int(targetH * 0.5)
    return CROP(image, w_center - w_delta, h_center - h_delta, w_center + w_delta, h_center + h_delta)

# TILING
def EDGEWRAP(image: cv2.Mat, tileX: float=1., tileY: float=1., edge: str="WRAP") -> cv2.Mat:
    height, width, _ = image.shape
    tileX = int(tileX * width * 0.5) if edge in ["WRAP", "WRAPX"] else 0
    tileY = int(tileY * height * 0.5) if edge in ["WRAP", "WRAPY"] else 0
    #print('EDGEWRAP', width, height, tileX, tileY)
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

# TRANSLATION
def TRANSLATE(image: cv2.Mat, offsetX: float, offsetY: float) -> cv2.Mat:
    height, width, _ = image.shape
    M = np.float32([[1, 0, offsetX * width], [0, 1, offsetY * height]])
    #print('TRANSLATE', offsetX, offsetY)
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

# ROTATION
def ROTATE(image: cv2.Mat, angle: float, center=(0.5 ,0.5)) -> cv2.Mat:
    height, width, _ = image.shape
    center = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    #print('ROTATE', angle)
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def SCALEFIT(image: cv2.Mat, width: int, height: int, mode: str="FIT") -> cv2.Mat:
    h, w, _ = image.shape
    if w == width and h == height:
        return image
    if mode == "ASPECT":
        scalar = max(width, height)
        scalar /= max(w, h)
        return cv2.resize(image, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_AREA)
    elif mode == "CROP":
        return CROP_CENTER(image, width, height)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Transform, Rotate and Scale followed by Tiling and then Inversion, conforming to an input wT, hT,
def TRANSFORM(image: cv2.Mat, offsetX: float=0., offsetY: float=0., angle: float=0., sizeX: float=1., sizeY: float=1., edge:str='CLIP', widthT: int=256, heightT: int=256, mode: str='FIX') -> cv2.Mat:
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
        #print('EDGEWRAP_POST', w, h)

    if sizeX != 1. or sizeY != 1.:
        wx = int(width * sizeX)
        hx = int(height * sizeY)
        #print('SCALE', wx, hx)
        image = cv2.resize(image, (wx, hx), interpolation=cv2.INTER_AREA)

    if edge != "CLIP":
        image = CROP_CENTER(image, width, height)
    #return image

    # TRANSLATION
    if offsetX != 0. or offsetY != 0.:
        if edge != "CLIP":
            image = EDGEWRAP(image)
        image = TRANSLATE(image, offsetX, offsetY)
        if edge != "CLIP":
            image = CROP_CENTER(image, width, height)

    # ROTATION
    if angle != 0:
        if edge != "CLIP":
            image = EDGEWRAP(image)
        image = ROTATE(image, angle)

    return SCALEFIT(image, widthT, heightT, mode=mode)

# INVERT
def INVERT(image: cv2.Mat, invert: float=1.) -> cv2.Mat:
    invert = min(max(invert, 0.), 1.)
    inverted = np.abs(255 - image)
    return cv2.addWeighted(image, 1. - invert, inverted, invert, 0)

def GAMMA(image, gamma):
    gamma_inv = 1. / max(0.01, min(0.9999999, gamma))
    return image.pow(gamma_inv)

def CONTRAST(image, contrast):
    image = (image - 0.5) * contrast + 0.5
    return torch.clamp(image, 0.0, 1.0)

def EXPOSURE(image, exposure):
    return image * (2.0**(exposure))

def HSV(image: cv2.Mat, hue, saturation, value):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue /= 360.
    image[:, :, 0] = (image[:, :, 0] + hue) % 180
    image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def MIRROR(image: cv2.Mat, pX: float, axis: int, invert: bool=False) -> cv2.Mat:
    output =  np.zeros_like(image)
    flip = cv2.flip(image, axis)
    height, width, _ = image.shape

    pX = min(max(pX, 0), 1)
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
