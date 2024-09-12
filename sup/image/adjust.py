"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Support
"""

from typing import Tuple

import cv2
import torch
import numpy as np

from loguru import logger

from Jovimetrix.sup.image import TYPE_IMAGE, EnumEdge, EnumInterpolation, \
    TYPE_fCOORD2D, bgr2image, cv2tensor, image2bgr, image_crop_center, tensor2cv

from Jovimetrix.sup.image.misc import image_histogram

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
    image: torch.tensor = cv2tensor(image)
    cc = image.shape[2]
    if cc == 4:
        old_alpha = image[..., 3]
        new_image = image[:, :, :3]
    elif cc == 1:
        new_image = np.repeat(image, 3, axis=2)
    else:
        new_image = image

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
