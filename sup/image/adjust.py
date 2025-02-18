"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Support
"""

from enum import Enum
from typing import List, Tuple

import cv2
import torch
import numpy as np

from . import \
    TYPE_IMAGE, TYPE_PIXEL, \
    TYPE_fCOORD2D, EnumImageType, \
    image_convert, image_mask_add, image_matte, image_minmax, bgr2image, \
    cv2tensor, image2bgr, tensor2cv

from .compose import image_blend, image_crop_center

from .channel import \
    EnumPixelSwizzle, \
    channel_solid

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumEdge(Enum):
    CLIP = 1
    WRAP = 2
    WRAPX = 3
    WRAPY = 4

class EnumInterpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4
    LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
    # INTER_MAX = cv2.INTER_MAX
    # WARP_FILL_OUTLIERS = cv2.WARP_FILL_OUTLIERS
    # WARP_INVERSE_MAP = cv2.WARP_INVERSE_MAP

class EnumMirrorMode(Enum):
    NONE = -1
    X = 0
    FLIP_X = 10
    Y = 20
    FLIP_Y = 30
    XY = 40
    X_FLIP_Y = 50
    FLIP_XY = 60
    FLIP_X_FLIP_Y = 70

class EnumScaleMode(Enum):
    # NONE = 0
    MATTE = 0
    CROP = 20
    FIT = 10
    ASPECT = 30
    ASPECT_SHORT = 35
    RESIZE_MATTE = 40

class EnumThreshold(Enum):
    BINARY = cv2.THRESH_BINARY
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO

class EnumThresholdAdapt(Enum):
    ADAPT_NONE = -1
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADAPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# ==============================================================================
# === IMAGE ===
# ==============================================================================

def image_contrast(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    image, alpha, cc = image2bgr(image)
    mean_value = np.mean(image)
    image = (image - mean_value) * value + mean_value
    image = np.clip(image, 0, 255).astype(np.uint8)
    return bgr2image(image, alpha, cc == 1)

def image_edge_wrap(image: TYPE_IMAGE, tileX: float=1., tileY: float=1.,
                    edge:EnumEdge=EnumEdge.WRAP) -> TYPE_IMAGE:
    """TILING."""
    height, width = image.shape[:2]
    tileX = int(width * tileX) if edge in [EnumEdge.WRAP, EnumEdge.WRAPX] else 0
    tileY = int(height * tileY) if edge in [EnumEdge.WRAP, EnumEdge.WRAPY] else 0
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

def image_equalize(image:TYPE_IMAGE) -> TYPE_IMAGE:
    image, alpha, cc = image2bgr(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return bgr2image(image, alpha, cc == 1)

def image_exposure(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    image, alpha, cc = image2bgr(image)
    image = np.clip(image * value, 0, 255).astype(np.uint8)
    return bgr2image(image, alpha, cc == 1)

def image_filter(image:TYPE_IMAGE, start:Tuple[int]=(128,128,128),
                 end:Tuple[int]=(128,128,128), fuzz:Tuple[float]=(0.5,0.5,0.5),
                 use_range:bool=False) -> Tuple[TYPE_IMAGE, TYPE_IMAGE]:
    """Filter an image based on a range threshold.
    It can use a start point with fuzziness factor and/or a start and end point with fuzziness on both points.

    Args:
        image (np.ndarray): Input image in the form of a NumPy array.
        start (tuple): The lower bound of the color range to be filtered.
        end (tuple): The upper bound of the color range to be filtered.
        fuzz (float): A factor for adding fuzziness (tolerance) to the color range.
        use_range (bool): Boolean indicating whether to use a start and end range or just the start point with fuzziness.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the filtered image and the mask.
    """
    old_alpha = None
    new_image = cv2tensor(image)
    cc = image.shape[2] if image.ndim > 2 else 1
    if cc == 4:
        old_alpha = new_image[..., 3]
        new_image = new_image[:, :, :3]
    elif cc == 1:
        if new_image.ndim == 2:
            new_image = new_image.unsqueeze(-1)
        new_image = torch.repeat_interleave(new_image, 3, dim=2)

    fuzz = torch.tensor(fuzz, dtype=torch.float64, device="cpu")
    start = torch.tensor(start, dtype=torch.float64, device="cpu") / 255.
    end = torch.tensor(end, dtype=torch.float64, device="cpu") / 255.
    if not use_range:
        end = start
    start -= fuzz
    end += fuzz
    start = torch.clamp(start, 0.0, 1.0)
    end = torch.clamp(end, 0.0, 1.0)

    mask = ((new_image[..., 0] > start[0]) & (new_image[..., 0] < end[0]))
    #mask |= ((new_image[..., 1] > start[1]) & (new_image[..., 1] < end[1]))
    #mask |= ((new_image[..., 2] > start[2]) & (new_image[..., 2] < end[2]))
    mask = ((new_image[..., 0] >= start[0]) & (new_image[..., 0] <= end[0]) &
            (new_image[..., 1] >= start[1]) & (new_image[..., 1] <= end[1]) &
            (new_image[..., 2] >= start[2]) & (new_image[..., 2] <= end[2]))

    output_image = torch.zeros_like(new_image)
    output_image[mask] = new_image[mask]

    if old_alpha is not None:
        output_image = torch.cat([output_image, old_alpha.unsqueeze(2)], dim=2)

    return tensor2cv(output_image), mask.cpu().numpy().astype(np.uint8) * 255

def image_flatten(image: List[TYPE_IMAGE], width:int=None, height:int=None,
                  mode=EnumScaleMode.MATTE,
                  sample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:

    if mode == EnumScaleMode.MATTE:
        width, height = image_minmax(image)[2:]
    else:
        h, w = image[0].shape[:2]
        width = width or w
        height = height or h

    current = np.zeros((height, width, 3), dtype=np.uint8)
    current = image_mask_add(current)
    for x in image:
        if mode != EnumScaleMode.MATTE and mode != EnumScaleMode.RESIZE_MATTE:
            x = image_scalefit(x, width, height, mode, sample)
        x = image_matte(x, (0,0,0,0), width, height)
        x = image_scalefit(x, width, height, EnumScaleMode.CROP, sample)
        x = image_convert(x, 4)
        #@TODO: ADD VARIOUS COMP OPS?
        current = cv2.add(current, x)
    return current

def image_gamma(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    # preserve original format
    image, alpha, cc = image2bgr(image)
    if value <= 0:
        image = (image * 0).astype(np.uint8)
    else:
        invGamma = 1.0 / max(0.000001, value)
        table = cv2.pow(np.arange(256) / 255, invGamma) * 255
        lookUpTable = np.clip(table, 0, 255).astype(np.uint8)
        image = cv2.LUT(image, lookUpTable)
        # now back to the original "format"
    return bgr2image(image, alpha, cc == 1)

def image_histogram(image:TYPE_IMAGE, bins=256) -> TYPE_IMAGE:
    bins = max(image.max(), bins) + 1
    flatImage = image.flatten()
    histogram = np.zeros(bins)
    for pixel in flatImage:
        histogram[pixel] += 1
    return histogram

def image_histogram_normalize(image:TYPE_IMAGE)-> TYPE_IMAGE:
    L = image.max()
    nonEqualizedHistogram = image_histogram(image, bins=L)
    sumPixels = np.sum(nonEqualizedHistogram)
    nonEqualizedHistogram = nonEqualizedHistogram/sumPixels
    cfdHistogram = np.cumsum(nonEqualizedHistogram)
    transformMap = np.floor((L-1) * cfdHistogram)
    flatNonEqualizedImage = list(image.flatten())
    flatEqualizedImage = [transformMap[p] for p in flatNonEqualizedImage]
    return np.reshape(flatEqualizedImage, image.shape)

def image_hsv(image: TYPE_IMAGE, hue: float, saturation: float, value: float) -> TYPE_IMAGE:
    image, alpha, cc = image2bgr(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue *= 255
    image[:, :, 0] = (image[:, :, 0] + hue) % 180
    image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return bgr2image(image, alpha, cc == 1)

def image_invert(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    """
    Invert an Grayscale, RGB or RGBA image using a specified inversion intensity.

    Parameters:
    - image: Input image as a NumPy array (RGB or RGBA).
    - value: Float between 0 and 1 representing the intensity of inversion (0: no inversion, 1: full inversion).

    Returns:
    - Inverted image.
    """
    # Clip the value to be within [0, 1] and scale to [0, 255]
    value = np.clip(value, 0, 1)
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]
        mask = alpha > 0
        inverted_rgb = 255 - rgb
        image = np.where(mask[:, :, None], (1 - value) * rgb + value * inverted_rgb, rgb)
        return np.dstack((image.astype(np.uint8), alpha))

    inverted_image = 255 - image
    return ((1 - value) * image + value * inverted_image).astype(np.uint8)

def image_mirror(image: TYPE_IMAGE, mode:EnumMirrorMode, x:float=0.5,
                 y:float=0.5) -> TYPE_IMAGE:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]

    def mirror(img:TYPE_IMAGE, axis:int, reverse:bool=False) -> TYPE_IMAGE:
        pivot = x if axis == 1 else y
        flip = cv2.flip(img, axis)
        pivot = np.clip(pivot, 0, 1)
        if reverse:
            pivot = 1. - pivot
            flip, img = img, flip

        scalar = height if axis == 0 else width
        slice1 = int(pivot * scalar)
        slice1w = scalar - slice1
        slice2w = min(scalar - slice1w, slice1w)

        if cc >= 3:
            output = np.zeros((height, width, cc), dtype=np.uint8)
        else:
            output = np.zeros((height, width), dtype=np.uint8)

        if axis == 0:
            output[:slice1, :] = img[:slice1, :]
            output[slice1:slice1 + slice2w, :] = flip[slice1w:slice1w + slice2w, :]
        else:
            output[:, :slice1] = img[:, :slice1]
            output[:, slice1:slice1 + slice2w] = flip[:, slice1w:slice1w + slice2w]

        return output

    if mode in [EnumMirrorMode.X, EnumMirrorMode.FLIP_X, EnumMirrorMode.XY, EnumMirrorMode.FLIP_XY, EnumMirrorMode.X_FLIP_Y, EnumMirrorMode.FLIP_X_FLIP_Y]:
        reverse = mode in [EnumMirrorMode.FLIP_X, EnumMirrorMode.FLIP_XY, EnumMirrorMode.FLIP_X_FLIP_Y]
        image = mirror(image, 1, reverse)

    if mode not in [EnumMirrorMode.NONE, EnumMirrorMode.X, EnumMirrorMode.FLIP_X]:
        reverse = mode in [EnumMirrorMode.FLIP_Y, EnumMirrorMode.FLIP_X_FLIP_Y, EnumMirrorMode.X_FLIP_Y]
        image = mirror(image, 0, reverse)

    return image

def image_pixelate(image: TYPE_IMAGE, amount:float=1.)-> TYPE_IMAGE:

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

def image_posterize(image: TYPE_IMAGE, levels:int=256) -> TYPE_IMAGE:
    divisor = 256 / max(2, min(256, levels))
    return (np.floor(image / divisor) * int(divisor)).astype(np.uint8)

def image_quantize(image:TYPE_IMAGE, levels:int=256, iterations:int=10,
                   epsilon:float=0.2) -> TYPE_IMAGE:
    levels = int(max(2, min(256, levels)))
    pixels = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    _, labels, centers = cv2.kmeans(pixels, levels, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(image.shape)

def image_rotate(image: TYPE_IMAGE, angle: float, center:TYPE_fCOORD2D=(0.5, 0.5),
                 edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    h, w = image.shape[:2]
    if edge != EnumEdge.CLIP:
        image = image_edge_wrap(image, edge=edge)

    height, width = image.shape[:2]
    c = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(c, -angle, 1.0)
    image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
    if edge != EnumEdge.CLIP:
        image = image_crop_center(image, w, h)
    return image

def image_scale(image: TYPE_IMAGE, scale:TYPE_fCOORD2D=(1.0, 1.0),
                sample:EnumInterpolation=EnumInterpolation.LANCZOS4,
                edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    h, w = image.shape[:2]
    if edge != EnumEdge.CLIP:
        image = image_edge_wrap(image, edge=edge)

    height, width = image.shape[:2]
    width = int(width * scale[0])
    height = int(height * scale[1])
    image = cv2.resize(image, (width, height), interpolation=sample.value)

    if edge != EnumEdge.CLIP:
        image = image_crop_center(image, w, h)
    return image

def image_scalefit(image: TYPE_IMAGE, width: int, height:int,
                mode:EnumScaleMode=EnumScaleMode.MATTE,
                sample:EnumInterpolation=EnumInterpolation.LANCZOS4,
                matte:TYPE_PIXEL=(0,0,0,0)) -> TYPE_IMAGE:

    match mode:
        case EnumScaleMode.MATTE:
            image = image_matte(image, matte, width, height)

        case EnumScaleMode.RESIZE_MATTE:
            canvas = np.full((height, width, 4), matte, dtype=image.dtype)
            image = image_blend(canvas, image)
            #image = image_matte(image, matte, width, height)

        case EnumScaleMode.ASPECT:
            h, w = image.shape[:2]
            ratio = max(width, height) / max(w, h)
            image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=sample.value)

        case EnumScaleMode.ASPECT_SHORT:
            h, w = image.shape[:2]
            ratio = min(width, height) / min(w, h)
            image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=sample.value)

        case EnumScaleMode.CROP:
            image = image_crop_center(image, width, height)

        case EnumScaleMode.FIT:
            image = cv2.resize(image, (width, height), interpolation=sample.value)

    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    return image

def image_sharpen(image:TYPE_IMAGE, kernel_size=None, sigma:float=1.0,
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

def image_swap_channels(imgA:TYPE_IMAGE, imgB:TYPE_IMAGE,
                        swap_in:Tuple[EnumPixelSwizzle, ...],
                        matte:Tuple[int,...]=(0,0,0,255)) -> TYPE_IMAGE:
    """Up-convert and swap all 4-channels of an image with another or a constant."""
    imgA = image_convert(imgA, 4)
    h,w = imgA.shape[:2]

    imgB = image_convert(imgB, 4)
    imgB = image_matte(imgB, (0,0,0,0), w, h)
    imgB = image_scalefit(imgB, w, h, EnumScaleMode.CROP)

    matte = (matte[2], matte[1], matte[0], matte[3])
    out = channel_solid(w, h, matte, EnumImageType.BGRA)
    swap_out = (EnumPixelSwizzle.RED_A,EnumPixelSwizzle.GREEN_A,
                EnumPixelSwizzle.BLUE_A,EnumPixelSwizzle.ALPHA_A)

    for idx, swap in enumerate(swap_in):
        if swap != EnumPixelSwizzle.CONSTANT:
            source_idx = swap.value // 10
            source_ab = swap.value % 10
            source = [imgA, imgB][source_ab]
            target_idx = swap_out[idx].value // 10
            out[:,:,target_idx] = source[:,:,source_idx]

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

def image_translate(image: TYPE_IMAGE, offset: TYPE_fCOORD2D=(0.0, 0.0),
                    edge: EnumEdge=EnumEdge.CLIP, border_value:int=0) -> TYPE_IMAGE:
    """
    Translates an image by a given offset. Supports various edge handling methods.

    Args:
        image (TYPE_IMAGE): Input image as a numpy array.
        offset (TYPE_fCOORD2D): Tuple (offset_x, offset_y) representing the translation offset.
        edge (EnumEdge): Enum representing edge handling method. Options are 'CLIP', 'WRAP', 'WRAPX', 'WRAPY'.

    Returns:
        TYPE_IMAGE: Translated image.
    """

    def translate(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        scalarX = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPX] else 1.0
        scalarY = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPY] else 1.0

        M = np.float32([[1, 0, offset[0] * width * scalarX], [0, 1, offset[1] * height * scalarY]])
        if edge == EnumEdge.CLIP:
            border_mode = cv2.BORDER_CONSTANT
        else:
            border_mode = cv2.BORDER_WRAP

        return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)

    return translate(image)

def image_transform(image: TYPE_IMAGE, offset:TYPE_fCOORD2D=(0.0, 0.0),
                    angle:float=0, scale:TYPE_fCOORD2D=(1.0, 1.0),
                    sample:EnumInterpolation=EnumInterpolation.LANCZOS4,
                    edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:
    sX, sY = scale
    if sX < 0:
        image = cv2.flip(image, 1)
        sX = -sX
    if sY < 0:
        image = cv2.flip(image, 0)
        sY = -sY
    if sX != 1. or sY != 1.:
        image = image_scale(image, (sX, sY), sample, edge)
    if angle != 0:
        image = image_rotate(image, angle, edge=edge)
    if offset[0] != 0. or offset[1] != 0.:
        image = image_translate(image, offset, edge)
    return image

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
