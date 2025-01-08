"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Channel Ops
"""

from enum import Enum
from typing import List

import numpy as np

from loguru import logger

from . import MIN_IMAGE_SIZE, TYPE_IMAGE, TYPE_PIXEL, \
    EnumImageType

from .color import pixel_eval

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumPixelSwizzle(Enum):
    RED_A = 0
    GREEN_A = 10
    BLUE_A = 20
    ALPHA_A = 30

    RED_B = 1
    GREEN_B = 11
    BLUE_B = 21
    ALPHA_B = 31
    CONSTANT = 52

# ==============================================================================
# === CHANNEL ===
# ==============================================================================

def channel_add(image:TYPE_IMAGE, color:TYPE_PIXEL=255) -> TYPE_IMAGE:
    """
    This function adds a new channel with a solid color to an image.

    :param image: The `image` parameter is expected to be an image represented as a
    NumPy array. The function assumes that the image has a shape attribute that
    returns a tuple representing the dimensions of the image (height, width, and
    channels if it's a color image)
    :type image: TYPE_IMAGE
    :param color: The `color` parameter in the `channel_add` function represents the
    color value that will be added as a new channel to the input image. The default
    value for `color` is 255, which is typically a white color in grayscale images,
    defaults to 255
    :type color: TYPE_PIXEL (optional)
    :return: The function `channel_add` returns a new image with an additional
    channel appended to the original image. The new channel has a solid color
    specified by the `color` parameter.
    """
    h, w = image.shape[:2]
    color = pixel_eval(color, EnumImageType.GRAYSCALE)
    new = channel_solid(w, h, color, EnumImageType.GRAYSCALE)
    return np.concatenate([image, new], axis=-1)

def channel_solid(width:int=MIN_IMAGE_SIZE, height:int=MIN_IMAGE_SIZE, color:TYPE_PIXEL=(0, 0, 0, 255),
                chan:EnumImageType=EnumImageType.BGR) -> TYPE_IMAGE:

    if chan == EnumImageType.GRAYSCALE:
        color = pixel_eval(color, EnumImageType.GRAYSCALE)
        what = np.full((height, width, 1), color, dtype=np.uint8)
        return what

    if not type(color) in [list, set, tuple]:
        color = [color]
    color += (0,) * (3 - len(color))
    if chan in [EnumImageType.BGR, EnumImageType.RGB]:
        if chan == EnumImageType.RGB:
            color = color[2::-1]
        return np.full((height, width, 3), color[:3], dtype=np.uint8)

    if len(color) < 4:
        color += (255,)

    if chan == EnumImageType.RGBA:
        color = color[2::-1]
    return np.full((height, width, 4), color, dtype=np.uint8)

def channel_merge(channels: List[TYPE_IMAGE]) -> TYPE_IMAGE:
    max_height = max(ch.shape[0] for ch in channels if ch is not None)
    max_width = max(ch.shape[1] for ch in channels if ch is not None)
    num_channels = len(channels)
    dtype = channels[0].dtype
    output = np.zeros((max_height, max_width, num_channels), dtype=dtype)

    for i, channel in enumerate(channels):
        if channel is None:
            continue

        h, w = channel.shape[:2]
        if channel.ndim > 2:
            channel = channel[..., 0]

        pad_top = (max_height - h) // 2
        pad_bottom = max_height - h - pad_top
        pad_left = (max_width - w) // 2
        pad_right = max_width - w - pad_left
        padded_channel = np.pad(channel, ((pad_top, pad_bottom), (pad_left, pad_right)),
                                mode='constant', constant_values=0)
        output[..., i] = padded_channel

    if num_channels == 1:
        output = output[..., 0]
    return output

def channel_swap(imageA:TYPE_IMAGE, swap_ot:EnumPixelSwizzle,
                imageB:TYPE_IMAGE, swap_in:EnumPixelSwizzle) -> TYPE_IMAGE:

    index_out = int(swap_ot.value / 10)
    cc_out = imageA.shape[2] if imageA.ndim == 3 else 1

    # swap channel is out of range of image size
    if index_out > cc_out:
        return imageA

    index_in = int(swap_in.value / 10)
    cc_in = imageB.shape[2] if imageB.ndim == 3 else 1
    if index_in > cc_in:
        return imageA

    img = imageA.copy()
    img[:,:,index_out] = imageB[:,:,index_in]
    return img
