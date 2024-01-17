"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition Support
"""

from enum import Enum
from typing import Optional

import cv2
import torch
import numpy as np
from scipy.ndimage import rotate
from blendmodes.blend import blendLayers, BlendType
from PIL import Image, ImageDraw
from loguru import logger

from Jovimetrix import TYPE_IMAGE, TYPE_PIXEL, TYPE_COORD

from Jovimetrix.sup.image import image_rgb_clean, image_rgb_restore, \
    image_grayscale, image_split, image_merge, channel_count, channel_add, \
    channel_solid, cv2pil, pil2cv, channel_fill, pixel_eval, \
    EnumScaleMode, EnumInterpolation

# =============================================================================
# === ENUM GLOBALS ===
# =============================================================================

class EnumThreshold(Enum):
    BINARY = cv2.THRESH_BINARY
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO

class EnumThresholdAdapt(Enum):
    ADAPT_NONE = -1
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADAPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

class EnumAdjustOP(Enum):
    BLUR = 0
    STACK_BLUR = 1
    GAUSSIAN_BLUR = 2
    MEDIAN_BLUR = 3
    SHARPEN = 10
    EMBOSS = 20
    # MEAN = 30 -- in UNARY
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
# === SHAPE FUNCTIONS ===
# =============================================================================

def shape_body(func: str, width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=(255,255,255), back:TYPE_PIXEL=(0,0,0)) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    d = ImageDraw.Draw(image)
    func = getattr(d, func)
    func(xy, fill=pixel_eval(fill))
    return image

def shape_ellipse(width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=(255,255,255), back:TYPE_PIXEL=(0,0,0)) -> Image:
    return shape_body('ellipse', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill, back=back)

def shape_quad(width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=(255,255,255), back:TYPE_PIXEL=(0,0,0)) -> Image:
    return shape_body('rectangle', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill, back=back)

def shape_polygon(width: int, height: int, size: float=1., sides: int=3, angle: float=0., fill:TYPE_PIXEL=(255,255,255), back:TYPE_PIXEL=(0,0,0)) -> Image:

    fill = pixel_eval(fill)
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), back)
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image

# =============================================================================
# === GEOMETRY FUNCTIONS ===
# =============================================================================

def geo_crop(image: TYPE_IMAGE,
             pnt_a: TYPE_COORD, pnt_b: TYPE_COORD,
             pnt_c: TYPE_COORD, pnt_d: TYPE_COORD,
             widthT: int=None, heightT: int=None,
             color: TYPE_PIXEL=0) -> TYPE_IMAGE:

        height, width = image.shape[:2]

        def process_point(pnt) -> TYPE_COORD:
            x, y = pnt
            x = np.clip(x, 0, 1) * width
            y = np.clip(y, 0, 1) * height
            return x, y

        x1, y1 = process_point(pnt_a)
        x2, y2 = process_point(pnt_b)
        x3, y3 = process_point(pnt_c)
        x4, y4 = process_point(pnt_d)

        x_max = max(x1, x2, x3, x4)
        x_min = min(x1, x2, x3, x4)
        y_max = max(y1, y2, y3, y4)
        y_min = min(y1, y2, y3, y4)

        x_start, x_end = int(max(0, x_min)), int(min(width, x_max))
        y_start, y_end = int(max(0, y_min)), int(min(height, y_max))

        crop_img = image[y_start:y_end, x_start:x_end]
        widthT = (widthT if widthT is not None else x_end - x_start)
        heightT = (heightT if heightT is not None else y_end - y_start)

        if (widthT == x_end - x_start and heightT == y_end - y_start):
            return crop_img

        cc = channel_count(image)[0]
        if isinstance(color, (float, int)):
            color = [color] * cc
        while len(color) > cc:
            color.pop(-1)

        if cc > 1:
            img_padded = np.full((heightT, widthT, cc), color, dtype=np.uint8)
        else:
            img_padded = np.full((heightT, widthT), color, dtype=np.uint8)

        crop_height, crop_width = crop_img.shape[:2]
        h2 = heightT // 2
        w2 = widthT // 2
        ch = crop_height // 2
        cw = crop_width // 2
        y_start, y_end = max(0, h2 - ch), min(h2 + ch, heightT)
        x_start, x_end = max(0, w2 - cw), min(w2 + cw, widthT)
        y_delta = (y_end - y_start) // 2
        x_delta = (x_end - x_start) // 2
        y_start2, y_end2 = int(max(0, ch - y_delta)), int(min(ch + y_delta, crop_height))
        x_start2, x_end2 = int(max(0, cw - x_delta)), int(min(cw + x_delta, crop_width))
        img_padded[y_start:y_end, x_start:x_end] = crop_img[y_start2:y_end2, x_start2:x_end2]
        return img_padded

def geo_crop_polygonal(image: TYPE_IMAGE,
                       pnt_a: TYPE_COORD, pnt_b: TYPE_COORD,
                       pnt_c: TYPE_COORD, pnt_d: TYPE_COORD,
                       color: TYPE_PIXEL=0) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:

    h, w = image.shape[:2]
    pnt_a = (int(pnt_a[0] * w), int(pnt_a[1] * h))
    pnt_b = (int(pnt_b[0] * w), int(pnt_b[1] * h))
    pnt_c = (int(pnt_c[0] * w), int(pnt_c[1] * h))
    pnt_d = (int(pnt_d[0] * w), int(pnt_d[1] * h))

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([pnt_a, pnt_b, pnt_c, pnt_d], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    canvas = np.full_like(image, color, dtype=image.dtype)
    canvas = cv2.bitwise_and(canvas, canvas, mask=~mask)
    roi = cv2.bitwise_and(image, image, mask=mask)
    img = cv2.addWeighted(canvas, 1, roi, 1, 0)
    img = img.astype(np.uint8)
    return img, mask

def geo_edge_wrap(image: TYPE_IMAGE, tileX: float=1., tileY: float=1., edge: str='WRAP') -> TYPE_IMAGE:
    """TILING."""
    height, width, _ = image.shape
    tileX = int(tileX * width * 0.5) if edge in ["WRAP", "WRAPX"] else 0
    tileY = int(tileY * height * 0.5) if edge in ["WRAP", "WRAPY"] else 0
    # logger.debug(f"[{width}, {height}]  [{tileX}, {tileY}]")
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

def geo_translate(image: TYPE_IMAGE, offsetX: float, offsetY: float) -> TYPE_IMAGE:
    """TRANSLATION."""
    height, width, _ = image.shape
    M = np.float32([[1, 0, offsetX * width], [0, 1, offsetY * height]])
    # logger.debug(f"[{offsetX}, {offsetY}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def geo_rotate(image: TYPE_IMAGE, angle: float, center:TYPE_COORD=(0.5 ,0.5)) -> TYPE_IMAGE:
    """ROTATION."""
    height, width, _ = image.shape
    center = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # logger.debug(f"[{angle}]")
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

def geo_scalefit(image: TYPE_IMAGE, width: int, height:int,
                 color:TYPE_PIXEL=0.,
                 mode:EnumScaleMode=EnumScaleMode.NONE,
                 sample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:

    # logger.debug("{} {} {} {}", mode, width, height, sample)

    match mode:
        case EnumScaleMode.ASPECT:
            h, w = image.shape[:2]
            aspect = min(width / w, height / h)
            return cv2.resize(image, None, fx=aspect, fy=aspect, interpolation=sample.value)

        case EnumScaleMode.CROP:
            return geo_crop(image, (0, 0), (0, 1), (1, 1), (1, 0), width, height, color)

        case EnumScaleMode.FIT:
            return cv2.resize(image, (width, height), interpolation=sample.value)

    return image

def geo_merge(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, axis: int=0, flip: bool=False) -> TYPE_IMAGE:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def geo_mirror(image: TYPE_IMAGE, pX: float, axis: int, invert: bool=False) -> TYPE_IMAGE:
    cc, _, width, height = channel_count(image)
    output =  np.zeros((height, width, 3), dtype=np.uint8)

    axis = 1 if axis == 0 else 0

    if cc > 3:
        alpha = image[:,:,3]
        alpha = cv2.flip(alpha, axis)
        image = image[:,:,:3]

    flip = cv2.flip(image, axis)

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

    if cc == 4:
        output = channel_add(output)
        output[:,:,3] = alpha

    return output

# =============================================================================
# === LIGHT FUNCTIONS ===
# =============================================================================

def light_hsv(image: TYPE_IMAGE, hue: float, saturation: float, value: float) -> TYPE_IMAGE:
    # preserve original format
    cc, image, alpha = image_rgb_clean(image)
    # work in RGB ==> HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue *= 255
    image[:, :, 0] = (image[:, :, 0] + hue) % 180
    image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # now back to the original "format"
    return image_rgb_restore(image, alpha, cc == 1)

def light_gamma(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    # preserve original format
    cc, image, alpha = image_rgb_clean(image)
    if value <= 0:
        image = (image * 0).astype(np.uint8)
    else:
        invGamma = 1.0 / max(0.000001, value)
        table = cv2.pow(np.arange(256) / 255.0, invGamma) * 255
        lookUpTable = np.clip(table, 0, 255).astype(np.uint8)
        image = cv2.LUT(image, lookUpTable)
        # now back to the original "format"
    return image_rgb_restore(image, alpha, cc == 1)

def light_contrast(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    cc, image, alpha = image_rgb_clean(image)
    mean_value = np.mean(image)
    image = (image - mean_value) * value + mean_value
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image_rgb_restore(image, alpha, cc == 1)

def light_exposure(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    cc, image, alpha = image_rgb_clean(image)
    image = np.clip(image * value, 0, 255).astype(np.uint8)
    return image_rgb_restore(image, alpha, cc == 1)

def light_invert(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    value = np.clip(value, 0, 1)
    cc, image, alpha = image_rgb_clean(image)
    image = cv2.addWeighted(image, 1 - value, 255 - image, value, 0)
    return image_rgb_restore(image, alpha, cc == 1)

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

def comp_blend(imageA:Optional[TYPE_IMAGE]=None,
               imageB:Optional[TYPE_IMAGE]=None,
               mask:Optional[TYPE_IMAGE]=None,
               blendOp:BlendType=BlendType.NORMAL,
               alpha:float=1.,
               color:TYPE_PIXEL=0.,
               mode:EnumScaleMode=EnumScaleMode.NONE,
               sample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:

    targetW = targetH = 0
    if mode == EnumScaleMode.NONE:
        targetW = max(imageA.shape[1] if imageA is not None else 0,
                      imageB.shape[1] if imageB is not None else 0)
        targetH = max(imageA.shape[0] if imageA is not None else 0,
                      imageB.shape[0] if imageB is not None else 0)
    else:
        targetW = imageA.shape[1] if imageA is not None else imageB.shape[1] \
            if imageB is not None else mask.shape[1] if mask is not None else 0
        targetH = imageA.shape[0] if imageA is not None else imageB.shape[0] \
            if imageB is not None else mask.shape[1] if mask is not None else 0
    imageB_maskColor = 0 if imageB is None else 255

    targetW, targetH = max(0, targetW), max(0, targetH)
    if targetH == 0 or targetW == 0:
        logger.warning("bad dimensions {} {}", targetW, targetH)
        return channel_solid(targetW or 1, targetH or 1, )

    def process(img:TYPE_IMAGE) -> TYPE_IMAGE:
        img = img if img is not None else channel_solid(targetW, targetH, 0)
        cc = channel_count(img)[0]
        while cc < 3:
            # @TODO: copy first channel to all missing? make grayscale RGB to process?
            img = channel_add(img, 0)
            cc += 1

        if cc < 4:
            img = channel_add(img, 255)

        img = geo_scalefit(img, targetW, targetH, color, mode, sample)
        h, w = img.shape[:2]
        if h != targetH or w != targetW:
            img = channel_fill(img, targetW, targetH, color)
        return img

    imageA = process(imageA)
    imageB = process(imageB)
    h, w = imageB.shape[:2]
    if mask is None:
        mask = np.full((h, w), imageB_maskColor, dtype=np.uint8)
    elif channel_count(mask)[0] != 1:
        mask = image_grayscale(mask)

    mH, mW = mask.shape[:2]
    if h != mH or w != mW:
        mask = geo_scalefit(mask, w, h, color, mode, sample)
        mask = channel_fill(mask, targetW, targetH, color)
        mask = np.squeeze(mask)

    imageB[:, :, 3] = mask[:, :]
    imageA = cv2pil(imageA)
    imageB = cv2pil(imageB)
    image = blendLayers(imageA, imageB, blendOp.value, np.clip(alpha, 0, 1))
    return pil2cv(image)

# =============================================================================
# === ADJUST FUNCTIONS ===
# =============================================================================

def adjust_equalize(image:TYPE_IMAGE) -> TYPE_IMAGE:
    cc, image, alpha = image_rgb_clean(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    return image_rgb_restore(image, alpha, cc == 1)

def adjust_threshold(image:TYPE_IMAGE, threshold:float=0.5,
                     mode:EnumThreshold=EnumThreshold.BINARY,
                     adapt:EnumThresholdAdapt=EnumThresholdAdapt.ADAPT_NONE,
                     block:int=3, const:float=0.) -> TYPE_IMAGE:

    const = max(-100, min(100, const))
    block = max(3, block if block % 2 == 1 else block + 1)
    cc, image, alpha = image_rgb_clean(image)
    if adapt != EnumThresholdAdapt.ADAPT_NONE:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, adapt.value, cv2.THRESH_BINARY, block, const)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # gray = np.stack([gray, gray, gray], axis=-1)
        image = cv2.bitwise_and(image, gray)
    else:
        threshold = int(threshold * 255)
        _, image = cv2.threshold(image, threshold, 255, mode.value)
    return image_rgb_restore(image, alpha, cc == 1)

def adjust_levels(image:torch.Tensor, black_point:int=0, white_point=255,
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

def adjust_sharpen(image:TYPE_IMAGE, kernel_size=None, sigma:float=1.0,
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

def adjust_quantize(image:TYPE_IMAGE, levels:int=256, iterations:int=10, epsilon:float=0.2) -> TYPE_IMAGE:
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
# === TEST ===
# =============================================================================

def testBlendModes() -> None:
    # all sizes and scale modes should work
    fore = cv2.imread('./_res/img/test_rainbow.png', cv2.IMREAD_UNCHANGED)
    fore = cv2.imread('./_res/img/test_fore2.png', cv2.IMREAD_UNCHANGED)
    back = cv2.imread('./_res/img/test_fore.png', cv2.IMREAD_UNCHANGED)
    # mask = cv2.imread('./_res/img/test_mask.png', cv2.IMREAD_UNCHANGED)
    mask = None
    for op in EnumBlendType:
        for m in EnumScaleMode:
            a = comp_blend(fore, fore, None, blendOp=op, mode=m, color=(255, 0, 0))
            cv2.imwrite(f'./_res/tst/blend-{op.name}-{m.name}-1.png', a)
            #a = comp_blend(back, None, mask, blendOp=op, alpha=0.5, color=(0, 255, 0), mode=m)
            #cv2.imwrite(f'./_res/tst/blend-{op.name}-{m.name}-0.5.png', a)
            #a = comp_blend(back, fore, None, blendOp=op, alpha=0, color=(0, 0, 255), mode=m)
            #cv2.imwrite(f'./_res/tst/blend-{op.name}-{m.name}-0.png', a)

def testImageMerge() -> None:
    img = cv2.imread('./_res/img/test_comfy.png', cv2.IMREAD_UNCHANGED)
    r, g, b, a = image_split(img)
    R = cv2.imread('./_res/img/test_R.png', cv2.IMREAD_UNCHANGED)
    G = cv2.imread('./_res/img/test_G.png', cv2.IMREAD_UNCHANGED)
    B = cv2.imread('./_res/img/test_B.png', cv2.IMREAD_UNCHANGED)
    d = image_merge(R, G, B, None, 512, 512)
    d = geo_scalefit(d, 512, 512)
    cv2.imwrite(f'./_res/tst/image-merge.png', d)

if __name__ == "__main__":
    a = cv2.imread('./res/img/color-a.png', cv2.IMREAD_UNCHANGED)
    b = cv2.imread('./res/img/test-a.png', cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('./res/img/mask-c.png', cv2.IMREAD_UNCHANGED)
    img = comp_blend(a, b, mask, mode=EnumScaleMode.CROP)
    cv2.imwrite(f'./_res/tst/image-blend.png', img)
    # testBlendModes()
    # testImageMerge()
