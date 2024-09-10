"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Image Support
"""

import io
import math
import base64
import sys
import urllib
import requests
from enum import Enum
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import cv2
import torch
import cupy as cp
import numpy as np
from numba import jit, cuda
from daltonlens import simulate
from scipy import ndimage
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageOps, ImageChops
from blendmodes.blend import BlendType, blendLayers

from loguru import logger

from Jovimetrix.sup.util import grid_make

# =============================================================================
# === GLOBAL ===
# =============================================================================

MIN_IMAGE_SIZE = 32
HALFPI = math.pi / 2
TAU = math.pi * 2

# =============================================================================
# === TYPE SHORTCUTS ===
# =============================================================================

TYPE_COORD = Union[
    Tuple[int, ...],
    Tuple[float, ...]
]

TYPE_PIXEL = Union[
    int,
    float,
    Tuple[float, ...],
    Tuple[int, ...],
]

TYPE_IMAGE = Union[np.ndarray, torch.Tensor]
TYPE_VECTOR = Union[TYPE_IMAGE | TYPE_PIXEL]

# =============================================================================
# === ENUM GLOBALS ===
# =============================================================================

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

class EnumColorTheory(Enum):
    COMPLIMENTARY = 0
    MONOCHROMATIC = 1
    SPLIT_COMPLIMENTARY = 2
    ANALOGOUS = 3
    TRIADIC = 4
    # TETRADIC = 5
    SQUARE = 6
    COMPOUND = 8
    # DOUBLE_COMPLIMENTARY = 9
    CUSTOM_TETRAD = 9

class EnumEdge(Enum):
    CLIP = 1
    WRAP = 2
    WRAPX = 3
    WRAPY = 4

class EnumGrayscaleCrunch(Enum):
    LOW = 0
    HIGH = 1
    MEAN = 2

class EnumImageType(Enum):
    GRAYSCALE = 0
    RGB = 10
    RGBA = 20
    BGR = 30
    BGRA = 40

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

class EnumIntFloat(Enum):
    FLOAT = 0
    INT = 1

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

class EnumOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    GRID = 2

class EnumProjection(Enum):
    NORMAL = 0
    POLAR = 5
    SPHERICAL = 10
    FISHEYE = 15
    PERSPECTIVE = 20

class EnumScaleMode(Enum):
    # NONE = 0
    MATTE = 0
    CROP = 20
    FIT = 10
    ASPECT = 30
    ASPECT_SHORT = 35

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

class EnumPixelSwizzle(Enum):
    RED_A = 20
    GREEN_A = 10
    BLUE_A = 0
    ALPHA_A = 30
    RED_B = 21
    GREEN_B = 11
    BLUE_B = 1
    ALPHA_B = 31
    CONSTANT = 50

class EnumCBSimulator(Enum):
    AUTOSELECT = 0
    BRETTEL1997 = 1
    COBLISV1 = 2
    COBLISV2 = 3
    MACHADO2009 = 4
    VIENOT1999 = 5
    VISCHECK = 6

class EnumCBDeficiency(Enum):
    PROTAN = simulate.Deficiency.PROTAN
    DEUTAN = simulate.Deficiency.DEUTAN
    TRITAN = simulate.Deficiency.TRITAN

# =============================================================================
# === CONVERSION GLOBAL ===
# =============================================================================

MODE_CV2 = {
    EnumImageType.BGRA: {
        4: cv2.COLOR_RGBA2BGRA,
        3: cv2.COLOR_RGB2BGRA,
        1: cv2.COLOR_GRAY2BGRA,
    },
    EnumImageType.RGBA: {
        4: lambda x: x,
        3: cv2.COLOR_RGB2RGBA,
        1: cv2.COLOR_GRAY2RGBA,
    },
    EnumImageType.BGR: {
        4: cv2.COLOR_RGBA2BGR,
        3: cv2.COLOR_RGB2BGR,
        1: cv2.COLOR_GRAY2BGR,
    },
    EnumImageType.RGB: {
        4: cv2.COLOR_RGBA2RGB,
        3: lambda x: x,
        1: cv2.COLOR_GRAY2RGB,
    },
    EnumImageType.GRAYSCALE: {
        4: cv2.COLOR_RGBA2GRAY,
        3: cv2.COLOR_RGB2GRAY,
        1: lambda x: x,
    }
}

# =============================================================================
# === COLOR SPACE CONVERSION ===
# =============================================================================

def gamma2linear(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Gamma correction for old PCs/CRT monitors"""
    return np.power(image, 2.2)

def linear2gamma(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Inverse gamma correction for old PCs/CRT monitors"""
    return np.power(np.clip(image, 0., 1.), 1.0 / 2.2)

def sRGB2Linear(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Convert sRGB to linearRGB, removing the gamma correction.
    Works for grayscale, RGB, or RGBA images.
    """
    image = image.astype(float) / 255.0

    # If the image has an alpha channel, separate it
    if image.shape[-1] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3]
    else:
        rgb = image
        alpha = None

    gamma = ((rgb + 0.055) / 1.055) ** 2.4
    scale = rgb / 12.92
    rgb = np.where(rgb > 0.04045, gamma, scale)

    # Recombine the alpha channel if it exists
    if alpha is not None:
        image = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1)
    else:
        image = rgb
    return (image * 255).astype(np.uint8)

def linear2sRGB(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Convert linearRGB to sRGB, applying the gamma correction.
    Works for grayscale, RGB, or RGBA images.
    """
    image = image.astype(float) / 255.0

    # If the image has an alpha channel, separate it
    if image.shape[-1] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3]
    else:
        rgb = image
        alpha = None

    higher = 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055
    lower = rgb * 12.92
    rgb = np.where(rgb > 0.0031308, higher, lower)

    # Recombine the alpha channel if it exists
    if alpha is not None:
        image = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1)
    else:
        image = rgb
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)

# =============================================================================
# === IMAGE CONVERSION ===
# =============================================================================

def bgr2hsv(bgr_color: TYPE_PIXEL) -> TYPE_PIXEL:
    return cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0, 0]

def bgr2image(image: TYPE_IMAGE, alpha: TYPE_IMAGE=None, gray: bool=False) -> TYPE_IMAGE:
    """Restore image with alpha, if any, and converting to grayscale (optional)."""
    if gray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_mask_add(image, alpha)

def b64_2_tensor(base64str: str) -> torch.Tensor:
    img = base64.b64decode(base64str)
    img = Image.open(BytesIO(img))
    img = ImageOps.exif_transpose(img)
    return pil2tensor(img)

def b64_2_pil(base64_string):
    prefix, base64_data = base64_string.split(",", 1)
    image_data = base64.b64decode(base64_data)
    image_stream = io.BytesIO(image_data)
    return Image.open(image_stream)

def b64_2_cv(base64_string) -> TYPE_IMAGE:
    _, data = base64_string.split(",", 1)
    data = base64.b64decode(data)
    data = io.BytesIO(data)
    data = Image.open(data)
    data = np.array(data)
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

def cv2pil(image: TYPE_IMAGE) -> Image.Image:
    """Convert a CV2 image to a PIL Image."""
    if (cc := image.shape[2] if len(image.shape) > 2 else 1) > 1:
        if cc == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(image)

def cv2tensor(image: TYPE_IMAGE, mask:bool=False) -> torch.Tensor:
    """Convert a CV2 image to a torch tensor."""
    if mask or image.ndim < 3 or (image.ndim == 3 and image.shape[2] == 1):
        mask = True
        image = image_grayscale(image)

    # image = linear2sRGB(image)
    image = image.astype(np.float32) / 255.0
    ret = torch.from_numpy(image).unsqueeze(0)
    if mask and ret.ndim == 4:
        ret = ret.squeeze(-1)
    return ret

def cv2tensor_full(image: TYPE_IMAGE, matte:TYPE_PIXEL=0) -> Tuple[torch.Tensor, ...]:
    rgba = image_convert(image, 4)
    rgb = image_matte(rgba, matte)[:,:,:3]
    mask = image_mask(rgba)
    rgba = torch.from_numpy(rgba.astype(np.float32) / 255.0).unsqueeze(0)
    rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
    return rgba, rgb, mask

def hsv2bgr(hsl_color: TYPE_PIXEL) -> TYPE_PIXEL:
    return cv2.cvtColor(np.uint8([[hsl_color]]), cv2.COLOR_HSV2BGR)[0, 0]

def image2bgr(image: TYPE_IMAGE) -> Tuple[TYPE_IMAGE, TYPE_IMAGE, int]:
    """RGB Helper function.
    Return channel count, BGR, and Alpha.
    """
    alpha = image_mask(image)
    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif cc == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image, alpha, cc

def pil2cv(image: Image.Image) -> TYPE_IMAGE:
    """Convert a PIL Image to a CV2 Matrix."""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a Torch Tensor."""
    image = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)

def tensor2cv(tensor: torch.Tensor) -> TYPE_IMAGE:
    """Convert a torch Tensor to a numpy ndarray."""
    tensor = tensor.cpu().squeeze().numpy()
    if tensor.ndim < 3:
        tensor = np.expand_dims(tensor, -1)
    image = np.clip(255.0 * tensor, 0, 255).astype(np.uint8)

    if image.shape[2] == 4:
        # image_flatten_mask
        mask = image_mask(image)
        # we should flatten against black?
        black = np.zeros(image.shape, dtype=np.uint8)
        image = image_blend(black, image, mask)
        image = image_mask_add(image, mask)
    return image

def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a torch Tensor to a PIL Image.
    Tensor should be HxWxC [no batch].
    """
    tensor = tensor.cpu().numpy().squeeze()
    tensor = np.clip(255. * tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

def mixlabLayer2cv(layer: dict) -> torch.Tensor:
    image=layer['image']
    mask=layer['mask']
    if 'type' in layer and layer['type']=='base64' and type(image) == str:
        image = b64_2_cv(image)
        mask = b64_2_cv(mask)
    else:
        image = tensor2cv(image)
        mask = tensor2cv(mask)
    return image_mask_add(image, mask)

# =============================================================================
# === PIXEL ===
# =============================================================================

def pixel_eval(color: TYPE_PIXEL,
            target: EnumImageType=EnumImageType.BGR,
            precision:EnumIntFloat=EnumIntFloat.INT,
            crunch:EnumGrayscaleCrunch=EnumGrayscaleCrunch.MEAN) -> Tuple[TYPE_PIXEL] | TYPE_PIXEL:
    """Evaluates R(GB)(A) pixels in range (0-255) into target target pixel type."""

    def parse_single_color(c: TYPE_PIXEL) -> TYPE_PIXEL:
        if not isinstance(c, int):
            c = np.clip(c, 0, 1)
            if precision == EnumIntFloat.INT:
                c = int(c * 255)
        else:
            c = np.clip(c, 0, 255)
            if precision == EnumIntFloat.FLOAT:
                c /= 255
        return c

    # make sure we are an RGBA value already
    if isinstance(color, (float, int)):
        color = tuple([parse_single_color(color)])
    elif isinstance(color, (set, tuple, list)):
        color = tuple([parse_single_color(c) for c in color])

    if target == EnumImageType.GRAYSCALE:
        alpha = 1
        if len(color) > 3:
            alpha = color[3]
            if precision == EnumIntFloat.INT:
                alpha /= 255
            color = color[:3]
        match crunch:
            case EnumGrayscaleCrunch.LOW:
                val = min(color) * alpha
            case EnumGrayscaleCrunch.HIGH:
                val = max(color) * alpha
            case EnumGrayscaleCrunch.MEAN:
                val = np.mean(color) * alpha
        if precision == EnumIntFloat.INT:
            val = int(val)
        return val

    if len(color) < 3:
        color += (0,) * (3 - len(color))

    if target in [EnumImageType.RGB, EnumImageType.BGR]:
        color = color[:3]
        if target == EnumImageType.BGR:
            color = color[::-1]
        return color

    if len(color) < 4:
        color += (255,)

    if target == EnumImageType.BGRA:
        color = tuple(color[2::-1]) + tuple([color[-1]])
    return color

def pixel_hsv_adjust(color:TYPE_PIXEL, hue:int=0, saturation:int=0, value:int=0, mod_color:bool=True, mod_sat:bool=False, mod_value:bool=False) -> TYPE_PIXEL:
    """Adjust an HSV type pixel.
    OpenCV uses... H: 0-179, S: 0-255, V: 0-255"""
    hsv = [0, 0, 0]
    hsv[0] = (color[0] + hue) % 180 if mod_color else np.clip(color[0] + hue, 0, 180)
    hsv[1] = (color[1] + saturation) % 255 if mod_sat else np.clip(color[1] + saturation, 0, 255)
    hsv[2] = (color[2] + value) % 255 if mod_value else np.clip(color[2] + value, 0, 255)
    return hsv

def pixel_convert(color:TYPE_PIXEL, size:int=4, alpha:int=255) -> TYPE_PIXEL:
    """
    This function converts X channel pixel into Y channel pixel by adjusting the
    size and alpha value if needed.

    :param color: The `color` parameter in the `pixel_convert` function represents
    the pixel value that you want to convert. It is expected to be a tuple
    representing the color channels of the pixel. The number of elements in the
    tuple should match the `size` parameter, which specifies the desired number of
    color channels
    :type color: TYPE_PIXEL
    :param size: The `size` parameter in the `pixel_convert` function specifies the
    number of channels in the pixel. It determines the expected size of the pixel
    tuple that is passed as the `color` argument. The function will modify the
    `color` tuple based on the `size` parameter to ensure it matches, defaults to 4
    :type size: int (optional)
    :param alpha: The `alpha` parameter in the `pixel_convert` function represents
    the alpha channel value of the pixel. It is an integer value ranging from 0 to
    255, where 0 indicates full transparency and 255 indicates full opacity. The
    default value for `alpha` is set to 255 if, defaults to 255
    :type alpha: int (optional)
    :return: The function `pixel_convert` returns the input `color` if its length is
    equal to the specified `size`. If the length of `color` is less than `size`, it
    pads the color with zeros to make it of the required length. If `size` is
    greater than 2, it adds alpha value to the color if `size` is 4. If `size` is
    """
    """Convert X channel pixel into Y channel pixel."""
    if (cc := len(color)) == size:
        return color
    if size > 2:
        color += (0,) * (3 - cc)
        if size == 4:
            color += (alpha,)
        return color
    return color[0]

# =============================================================================
# === CHANNEL ===
# =============================================================================

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
    h, w = imageA.shape[:2]
    imageB = image_scalefit(imageB, w, h, EnumScaleMode.FIT)
    imageA[:,:,index_out] = imageB[:,:,index_in]
    return imageA

# =============================================================================
# === EXPLICIT SHAPE FUNCTIONS ===
# =============================================================================

def shape_ellipse(width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    ImageDraw.Draw(image).ellipse(xy, fill=fill)
    return image

def shape_quad(width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    ImageDraw.Draw(image).rectangle(xy, fill=fill)
    return image

def shape_polygon(width: int, height: int, size: float=1., sides: int=3, fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), back)
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image

# =============================================================================
# === IMAGE ===
# =============================================================================

def image_affine_edge(image: TYPE_IMAGE, callback: object, edge: EnumEdge=EnumEdge.WRAP) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    if edge != EnumEdge.CLIP:
        image = image_edge_wrap(image, edge=edge)
    image = callback(image)
    image = image_crop_center(image, width, height)
    return image

def image_blend(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, mask:Optional[TYPE_IMAGE]=None,
                blendOp:BlendType=BlendType.NORMAL, alpha:float=1) -> TYPE_IMAGE:

    h, w = imageA.shape[:2]
    imageA = image_convert(imageA, 4)
    imageA = cv2pil(imageA)
    imageB = image_convert(imageB, 4)
    h2, w2 = imageB.shape[:2]
    w2 = min(w, w2)
    h2 = min(h, h2)
    imageB = image_crop_center(imageB, w2, h2)
    imageB = image_matte(imageB, (0,0,0,0), w, h)
    imageB = image_convert(imageB, 4)
    #
    #old_mask = image_mask(imageB)
    #if len(old_mask.shape) > 2:
    #    old_mask = old_mask[..., 0][:,:]

    if mask is not None:
        mask = image_crop_center(mask, w, h)
        mask = image_matte(mask, (0,0,0,0), w, h)
        if len(mask.shape) > 2:
            mask = mask[..., 0][:,:]
        #old_mask = cv2.bitwise_and(mask, old_mask)

    imageB[..., 3] = mask #old_mask
    imageB = cv2pil(imageB)
    alpha = np.clip(alpha, 0, 1)
    image = blendLayers(imageA, imageB, blendOp.value, alpha)
    image = pil2cv(image)

    #if mask is not None:
    #    image = image_mask_add(image, mask)

    return image_crop_center(image, w, h)

def image_by_size(image_list: List[TYPE_IMAGE], enumSize: EnumImageBySize=EnumImageBySize.LARGEST) -> Tuple[TYPE_IMAGE, int, int]:

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

def image_color_blind(image: TYPE_IMAGE, deficiency:EnumCBDeficiency,
                    simulator:EnumCBSimulator=EnumCBSimulator.AUTOSELECT,
                    severity:float=1.0) -> TYPE_IMAGE:

    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 4:
        mask = image_mask(image)
    image = image_convert(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    match simulator:
        case EnumCBSimulator.AUTOSELECT:
            simulator = simulate.Simulator_AutoSelect()
        case EnumCBSimulator.BRETTEL1997:
            simulator = simulate.Simulator_Brettel1997()
        case EnumCBSimulator.COBLISV1:
            simulator = simulate.Simulator_CoblisV1()
        case EnumCBSimulator.COBLISV2:
            simulator = simulate.Simulator_CoblisV2()
        case EnumCBSimulator.MACHADO2009:
            simulator = simulate.Simulator_Machado2009()
        case EnumCBSimulator.VIENOT1999:
            simulator = simulate.Simulator_Vienot1999()
        case EnumCBSimulator.VISCHECK:
            simulator = simulate.Simulator_Vischeck()
    image = simulator.simulate_cvd(image, deficiency.value, severity=severity)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if cc == 4:
        image = image_mask_add(image, mask)
    return image

def image_contrast(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    image, alpha, cc = image2bgr(image)
    mean_value = np.mean(image)
    image = (image - mean_value) * value + mean_value
    image = np.clip(image, 0, 255).astype(np.uint8)
    return bgr2image(image, alpha, cc == 1)

def image_convert(image: TYPE_IMAGE, channels: int) -> TYPE_IMAGE:
    """Force image format to a specific number of channels.

    Args:
        image (TYPE_IMAGE): Input image.
        channels (int): Desired number of channels (1, 3, or 4).

    Returns:
        TYPE_IMAGE: Image with the specified number of channels.
    """
    if image.ndim == 2:
        # Expand grayscale image to have a channel dimension
        image = np.expand_dims(image, -1)

    # Ensure channels value is within the valid range
    channels = max(1, min(4, channels))
    cc = image.shape[2] if image.ndim == 3 else 1

    if cc == channels:
        return image

    if channels == 1:
        return image_grayscale(image)

    if channels == 3:
        if cc == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif cc == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if cc == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

def image_crop_polygonal(image: TYPE_IMAGE, points: List[TYPE_COORD]) -> TYPE_IMAGE:
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

def image_crop_head(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """
    Given a file path or np.ndarray image with a face,
    returns cropped np.ndarray around the largest detected
    face.

    Parameters
    ----------
    - `path_or_array` : {`str`, `np.ndarray`}
        * The filepath or numpy array of the image.

    Returns
    -------
    - `image` : {`np.ndarray`, `None`}
        * A cropped numpy array if face detected, else None.
    """

    MIN_FACE = 8

    gray = image_grayscale(image)
    h, w = image.shape[:2]
    minface = int(np.sqrt(h**2 + w**2) / MIN_FACE)

    '''
        # Create the haar cascade
        face_cascade = cv2.CascadeClassifier(self.casc_path)

        # ====== Detect faces in the image ======
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(minface, minface),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,
        )

        # Handle no faces
        if len(faces) == 0:
            return None

        # Make padding from biggest face found
        x, y, w, h = faces[-1]
        pos = self._crop_positions(
            img_height,
            img_width,
            x,
            y,
            w,
            h,
        )

        # ====== Actual cropping ======
        image = image[pos[0] : pos[1], pos[2] : pos[3]]

        # Resize
        if self.resize:
            with Image.fromarray(image) as img:
                image = np.array(img.resize((self.width, self.height)))

        # Underexposition fix
        if self.gamma:
            image = check_underexposed(image, gray)
        return bgr_to_rbg(image)

    def _determine_safe_zoom(self, imgh, imgw, x, y, w, h):
        """
        Determines the safest zoom level with which to add margins
        around the detected face. Tries to honor `self.face_percent`
        when possible.

        Parameters:
        -----------
        imgh: int
            Height (px) of the image to be cropped
        imgw: int
            Width (px) of the image to be cropped
        x: int
            Leftmost coordinates of the detected face
        y: int
            Bottom-most coordinates of the detected face
        w: int
            Width of the detected face
        h: int
            Height of the detected face

        Diagram:
        --------
        i / j := zoom / 100

                  +
        h1        |         h2
        +---------|---------+
        |      MAR|GIN      |
        |         (x+w, y+h)|
        |   +-----|-----+   |
        |   |   FA|CE   |   |
        |   |     |     |   |
        |   ├──i──┤     |   |
        |   |  cen|ter  |   |
        |   |     |     |   |
        |   +-----|-----+   |
        |   (x, y)|         |
        |         |         |
        +---------|---------+
        ├────j────┤
                  +
        """
        # Find out what zoom factor to use given self.aspect_ratio
        corners = itertools.product((x, x + w), (y, y + h))
        center = np.array([x + int(w / 2), y + int(h / 2)])
        i = np.array(
            [(0, 0), (0, imgh), (imgw, imgh), (imgw, 0), (0, 0)]
        )  # image_corners
        image_sides = [(i[n], i[n + 1]) for n in range(4)]

        corner_ratios = [self.face_percent]  # Hopefully we use this one
        for c in corners:
            corner_vector = np.array([center, c])
            a = distance(*corner_vector)
            intersects = list(intersect(corner_vector, side) for side in image_sides)
            for pt in intersects:
                if (pt >= 0).all() and (pt <= i[2]).all():  # if intersect within image
                    dist_to_pt = distance(center, pt)
                    corner_ratios.append(100 * a / dist_to_pt)
        return max(corner_ratios)

    def _crop_positions(
        self,
        imgh,
        imgw,
        x,
        y,
        w,
        h,
    ):
        """
        Retuns the coordinates of the crop position centered
        around the detected face with extra margins. Tries to
        honor `self.face_percent` if possible, else uses the
        largest margins that comply with required aspect ratio
        given by `self.height` and `self.width`.

        Parameters:
        -----------
        imgh: int
            Height (px) of the image to be cropped
        imgw: int
            Width (px) of the image to be cropped
        x: int
            Leftmost coordinates of the detected face
        y: int
            Bottom-most coordinates of the detected face
        w: int
            Width of the detected face
        h: int
            Height of the detected face
        """
        zoom = self._determine_safe_zoom(imgh, imgw, x, y, w, h)

        # Adjust output height based on percent
        if self.height >= self.width:
            height_crop = h * 100.0 / zoom
            width_crop = self.aspect_ratio * float(height_crop)
        else:
            width_crop = w * 100.0 / zoom
            height_crop = float(width_crop) / self.aspect_ratio

        # Calculate padding by centering face
        xpad = (width_crop - w) / 2
        ypad = (height_crop - h) / 2

        # Calc. positions of crop
        h1 = x - xpad
        h2 = x + w + xpad
        v1 = y - ypad
        v2 = y + h + ypad

        return [int(v1), int(v2), int(h1), int(h2)]
    '''

def image_detect(image: TYPE_IMAGE) -> Tuple[TYPE_IMAGE, Tuple[int, ...]]:
    gray = image_grayscale(image)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the item we want to recenter
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image, (x, y, w, h)

def image_diff(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, threshold:int=0, color:TYPE_PIXEL=(255, 0, 0)) -> Tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, float]:
    """imageA, imageB, diff, thresh, score
    """
    h1, w1 = imageA.shape[:2]
    h2, w2 = imageB.shape[:2]
    w1 = max(w1, w2)
    h1 = max(h1, h2)
    imageA = image_matte(imageA, (0, 0, 0, 0), w1, h1)
    imageA = image_convert(imageA, 3)
    imageB = image_matte(imageB, (0, 0, 0, 0), w1, h1)
    imageB = image_convert(imageB, 3)
    grayA = image_grayscale(imageA)
    grayB = image_grayscale(imageB)
    (score, diff) = ssim(grayA, grayB, full=True, channel_axis=2)
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    high_a = imageA.copy()
    high_a = image_convert(high_a, 3)
    high_b = imageB.copy()
    high_b = image_convert(high_b, 3)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(high_a, [c], 0, color[::-1], -1)
        cv2.drawContours(high_b, [c], 0, color[::-1], -1)
        cv2.drawContours(diff_box, [c], 0, color[::-1], -1)
    imageA = cv2.addWeighted(imageA, 0.0, high_a, 1, 0)
    imageB = cv2.addWeighted(imageB, 0.0, high_b, 1, 0)
    return imageA, imageB, diff, thresh, score

def image_disparity(imageA: np.ndarray) -> np.ndarray:
    imageA = imageA.astype(np.float32) / 255.
    imageA = cv2.normalize(imageA, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    disparity_map = np.divide(1.0, imageA, where=imageA != 0)
    return np.where(imageA == 0, 1, disparity_map)

def image_edge_wrap(image: TYPE_IMAGE, tileX: float=1., tileY: float=1., edge:EnumEdge=EnumEdge.WRAP) -> TYPE_IMAGE:
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

def image_filter(image:TYPE_IMAGE, start:Tuple[int]=(128,128,128), end:Tuple[int]=(128,128,128), fuzz:Tuple[float]=(0.5,0.5,0.5), use_range:bool=False) -> Tuple[TYPE_IMAGE, TYPE_IMAGE]:
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

def image_flatten_mask(image:TYPE_IMAGE, matte:Tuple=(0,0,0,255)) -> Tuple[TYPE_IMAGE, TYPE_IMAGE|None]:
    """Flatten the image with its own alpha channel, if any."""
    mask = image_mask(image)
    return image_blend(image, image, mask), mask

def image_formats() -> List[str]:
    exts = Image.registered_extensions()
    return [ex for ex, f in exts.items() if f in Image.OPEN]

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

def image_gradient_map2(image, gradient_map):
    na = np.array(image)
    grey = np.mean(na, axis=2).astype(np.uint8)
    cmap = np.array(gradient_map.convert('RGB'))
    result = np.zeros((*grey.shape, 3), dtype=np.uint8)
    grey_reshaped = grey.reshape(-1)
    np.take(cmap.reshape(-1, 3), grey_reshaped, axis=0, out=result.reshape(-1, 3))
    return result

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

    def pixel(x, spread:int=1) -> Tuple[int, int, int]:
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

# Adapted from WAS Suite -- gradient_map
# https://github.com/WASasquatch/was-node-suite-comfyui
def image_gradient_map(image:TYPE_IMAGE, gradient_map:TYPE_IMAGE, reverse:bool=False) -> TYPE_IMAGE:
    if reverse:
        gradient_map = gradient_map[:,:,::-1]
    grey = image_grayscale(image)
    cmap = image_convert(gradient_map, 3)
    cmap = cv2.resize(cmap, (256, 256))
    cmap = cmap[0,:,:].reshape((256, 1, 3)).astype(np.uint8)
    return cv2.applyColorMap(grey, cmap)

def image_grayscale(image: TYPE_IMAGE, use_alpha: bool = False) -> TYPE_IMAGE:
    """Convert image to grayscale, optionally using the alpha channel if present.

    Args:
        image (TYPE_IMAGE): Input image, potentially with multiple channels.
        use_alpha (bool): If True and the image has 4 channels, multiply the grayscale
                          values by the alpha channel. Defaults to False.

    Returns:
        TYPE_IMAGE: Grayscale image, optionally alpha-multiplied.
    """
    if image.ndim == 2 or image.shape[2] == 1:
        return image

    if image.shape[2] == 4:
        # Convert RGBA to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if use_alpha:
            # Normalize alpha to [0, 1]
            alpha_channel = image[:, :, 3] / 255.0
            grayscale = (grayscale * alpha_channel).astype(np.uint8)
        return grayscale

    # Convert RGB to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def image_grid(data: List[TYPE_IMAGE], width: int, height: int) -> TYPE_IMAGE:
    #@TODO: makes poor assumption all images are the same dimensions.
    chunks, col, row = grid_make(data)
    frame = np.zeros((height * row, width * col, 4), dtype=np.uint8)
    i = 0
    for y, strip in enumerate(chunks):
        for x, item in enumerate(strip):
            cc = item.shape[2] if item.ndim == 3 else 1
            if cc == 3:
                item = channel_add(item)
            y1, y2 = y * height, (y+1) * height
            x1, x2 = x * width, (x+1) * width
            frame[y1:y2, x1:x2, ] = item
            i += 1

    return frame

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

def image_histogram_statistics(histogram:np.ndarray, L=256)-> TYPE_IMAGE:
    sumPixels = np.sum(histogram)
    normalizedHistogram = histogram/sumPixels
    mean = 0
    for i in range(L):
        mean += i * normalizedHistogram[i]
    variance = 0
    for i in range(L):
        variance += (i-mean)**2 * normalizedHistogram[i]
    std = np.sqrt(variance)
    return mean, variance, std

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

def image_lerp(imageA: TYPE_IMAGE, imageB:TYPE_IMAGE, mask:TYPE_IMAGE=None,
               alpha:float=1.) -> TYPE_IMAGE:

    imageA = imageA.astype(np.float32)
    imageB = imageB.astype(np.float32)

    # establish mask
    alpha = np.clip(alpha, 0, 1)
    if mask is None:
        height, width = imageA.shape[:2]
        mask = np.ones((height, width, 1), dtype=np.float32)
    else:
        # normalize the mask
        mask = mask.astype(np.float32)
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * alpha

    # LERP
    imageA = cv2.multiply(1. - mask, imageA)
    imageB = cv2.multiply(mask, imageB)
    imageA = (cv2.add(imageA, imageB) / 255. - 0.5) * 2.0
    imageA = (imageA * 255).astype(np.uint8)
    return np.clip(imageA, 0, 255)

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

def image_load(url: str) -> Tuple[TYPE_IMAGE, ...]:
    try:
        img = cv2.imread(url, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"{url} could not be loaded.")

        img = image_normalize(img)
        logger.debug(f"load image {url}: {img.ndim} {img.shape}")
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim < 3:
            img = np.expand_dims(img, axis=2)

    except Exception:
        logger.debug(f"load image fallback to PIL {url}")
        try:
            img = Image.open(url)
            img = ImageOps.exif_transpose(img)
            img = np.array(img)
            if img.dtype != np.uint8:
                img = np.clip(np.array(img * 255), 0, 255).astype(dtype=np.uint8)
        except Exception as e:
            logger.error(str(e))
            raise Exception(f"Error loading image: {e}")

    if img is None:
        raise Exception(f"No file found at {url}")

    mask = image_mask(img)
    img = image_blend(img, img, mask)
    return img, mask

def image_load_data(data: str) -> TYPE_IMAGE:
    img = ImageOps.exif_transpose(data)
    return pil2cv(img)

def image_load_exr(url: str) -> Tuple[TYPE_IMAGE, TYPE_IMAGE]:
    """
    exr_file     = OpenEXR.InputFile(url)
    exr_header   = exr_file.header()
    r,g,b = exr_file.channels("RGB", pixel_type=Imath.PixelType(Imath.PixelType.FLOAT) )

    dw = exr_header[ "dataWindow" ]
    w  = dw.max.x - dw.min.x + 1
    h  = dw.max.y - dw.min.y + 1

    image = np.ones( (h, w, 4), dtype = np.float32 )
    image[:, :, 0] = np.core.multiarray.frombuffer( r, dtype = np.float32 ).reshape(h, w)
    image[:, :, 1] = np.core.multiarray.frombuffer( g, dtype = np.float32 ).reshape(h, w)
    image[:, :, 2] = np.core.multiarray.frombuffer( b, dtype = np.float32 ).reshape(h, w)
    return create_optix_image_2D( w, h, image.flatten() )
    """
    pass

def image_load_from_url(url: str) -> TYPE_IMAGE:
    """Creates a CV2 BGR image from a url."""
    try:
        image  = urllib.request.urlopen(url)
        image = np.asarray(bytearray(image.read()), dtype=np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            return pil2cv(image)
        except Exception as e:
            logger.error(str(e))

def image_mask(image:TYPE_IMAGE, color:TYPE_PIXEL=255) -> TYPE_IMAGE:
    """Create a mask from the image, preserving transparency."""
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., 3]
    image = np.ones_like(image, dtype=np.uint8) * color
    return image[:,:,0]

def image_mask_add(image:TYPE_IMAGE, mask:TYPE_IMAGE=None, alpha:float=255) -> TYPE_IMAGE:
    """Put custom mask into an image. If there is no mask, alpha is applied.
    Images are expanded to 4 channels.
    Existing 4 channel images with no mask input just return themselves.
    """
    image = image_convert(image, 4)
    mask = image_mask(image, alpha) if mask is None else image_convert(mask, 1)
    image[..., 3] = mask if mask.ndim == 2 else mask[:, :, 0]
    return image

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

def image_matte(image: TYPE_IMAGE, color: Tuple[int,int,int,int]=(0,0,0,255), width: int=None, height: int=None) -> TYPE_IMAGE:
    """
    Puts an image atop a colored matte with the same dimensions as the image.

    Args:
        image (TYPE_IMAGE): The input image.
        color (tuple(int, int, int, int)): The color of the matte as a tuple (R, G, B, A).
        width (int, optional): The width of the matte. Defaults to the image width.
        height (int, optional): The height of the matte. Defaults to the image height.

    Returns:
        TYPE_IMAGE: The composited image on a matte. Output is RGBA with the original Alpha (if any) or solid white.
    """
    # Determine the dimensions of the image and the matte
    image_height, image_width = image.shape[:2]
    width = width or image_width
    height = height or image_height

    # Create a solid matte with the specified color
    matte = np.full((height, width, 4), color, dtype=np.uint8)

    # Ensure the image has 4 channels (RGBA)
    image = image_convert(image, 4)

    # Extract the alpha channel from the image
    alpha = image[:, :, 3] / 255.0

    # Calculate the center position for the image on the matte
    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    # Place the image onto the matte using the alpha channel for blending
    for c in range(0, 3):
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] = \
            (1 - alpha) * matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] + \
            alpha * image[:, :, c]

    # Set the alpha channel of the matte to the maximum of the matte's and the image's alpha
    matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3] = \
        np.maximum(matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3], image[:, :, 3])

    return matte

def image_merge(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, axis: int=0, flip: bool=False) -> TYPE_IMAGE:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def image_minmax(image:List[TYPE_IMAGE]) -> Tuple[int, int, int, int]:
    h_min = w_min = 100000000000
    h_max = w_max = MIN_IMAGE_SIZE
    for img in image:
        if img is None:
            continue
        h, w = img.shape[:2]
        h_max = max(h, h_max)
        w_max = max(w, w_max)
        h_min = min(h, h_min)
        w_min = min(w, w_min)

    # x,y - x+width, y+height
    return w_min, h_min, w_max, h_max

def image_mirror(image: TYPE_IMAGE, mode:EnumMirrorMode, x:float=0.5, y:float=0.5) -> TYPE_IMAGE:
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

def image_mirror_mandela(imageA: np.ndarray, imageB: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Merge 4 flipped copies of input images to make them wrap.
    Output is twice bigger in both dimensions."""

    top = np.hstack([imageA, -np.flip(imageA, axis=1)])
    bottom = np.hstack([np.flip(imageA, axis=0), -np.flip(imageA)])
    imageA = np.vstack([top, bottom])

    top = np.hstack([imageB, np.flip(imageB, axis=1)])
    bottom = np.hstack([-np.flip(imageB, axis=0), -np.flip(imageB)])
    imageB = np.vstack([top, bottom])
    return imageA, imageB

def image_normalize(image: TYPE_IMAGE) -> TYPE_IMAGE:
    image = image.astype(np.float32)
    img_min = np.min(image)
    img_max = np.max(image)
    if img_min == img_max:
        return np.zeros_like(image)
    image = (image - img_min) / (img_max - img_min)
    return (image * 255).astype(np.uint8)

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

def image_quantize(image:TYPE_IMAGE, levels:int=256, iterations:int=10, epsilon:float=0.2) -> TYPE_IMAGE:
    levels = int(max(2, min(256, levels)))
    pixels = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    _, labels, centers = cv2.kmeans(pixels, levels, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(image.shape)

def image_recenter(image: TYPE_IMAGE) -> TYPE_IMAGE:
    cropped_image = image_detect(image)[0]
    new_image = np.zeros(image.shape, dtype=np.uint8)
    paste_x = (new_image.shape[1] - cropped_image.shape[1]) // 2
    paste_y = (new_image.shape[0] - cropped_image.shape[0]) // 2
    new_image[paste_y:paste_y+cropped_image.shape[0], paste_x:paste_x+cropped_image.shape[1]] = cropped_image
    return new_image

def image_rotate(image: TYPE_IMAGE, angle: float, center:TYPE_COORD=(0.5, 0.5), edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    def func_rotate(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        c = (int(width * center[0]), int(height * center[1]))
        M = cv2.getRotationMatrix2D(c, -angle, 1.0)
        return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR)

    image = image_affine_edge(image, func_rotate, edge)
    return image

def image_save_gif(fpath:str, images: List[Image.Image], fps: int=0,
                loop:int=0, optimize:bool=False) -> None:

    fps = min(50, max(1, fps))
    images[0].save(
        fpath,
        append_images=images[1:],
        duration=3, # int(100.0 / fps),
        loop=loop,
        optimize=optimize,
        save_all=True
    )

def image_scale(image: TYPE_IMAGE, scale:TYPE_COORD=(1.0, 1.0), sample:EnumInterpolation=EnumInterpolation.LANCZOS4, edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    def scale_func(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        width = int(width * scale[0])
        height = int(height * scale[1])
        return cv2.resize(img, (width, height), interpolation=sample.value)

    if edge == EnumEdge.CLIP:
        return scale_func(image)
    return image_affine_edge(image, scale_func, edge)

def image_scalefit(image: TYPE_IMAGE, width: int, height:int,
                mode:EnumScaleMode=EnumScaleMode.MATTE,
                sample:EnumInterpolation=EnumInterpolation.LANCZOS4,
                matte:TYPE_PIXEL=(0,0,0,0)) -> TYPE_IMAGE:

    match mode:
        case EnumScaleMode.MATTE:
            image = image_matte(image, matte, width, height)

        case EnumScaleMode.ASPECT:
            h, w = image.shape[:2]
            ratio = max(width, height) / max(w, h)
            image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=sample.value)

        case EnumScaleMode.ASPECT_SHORT:
            h, w = image.shape[:2]
            ratio = min(width, height) / min(w, h)
            image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=sample.value)

        case EnumScaleMode.CROP:
            image = cv2.resize(image_crop_center(image, width, height), (width, height))

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

def image_split(image: TYPE_IMAGE, convert:object=image_grayscale) -> Tuple[TYPE_IMAGE, ...]:
    h, w = image.shape[:2]
    dtype = image.dtype

    # Grayscale image
    if image.ndim == 2 or image.shape[2] == 1:
        r = g = b = image.reshape(h, w)
        a = np.full((h, w), 255, dtype=dtype)

    # BGR image
    elif image.shape[2] == 3:
        r, g, b = cv2.split(image)
        a = np.full((h, w), 255, dtype=dtype)
    else:
        r, g, b, a = cv2.split(image)
    return r, g, b, a

def image_stack(image_list: List[TYPE_IMAGE], axis:EnumOrientation=EnumOrientation.HORIZONTAL,
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

def image_stereogram(image: TYPE_IMAGE, depth: TYPE_IMAGE, divisions:int=8, mix:float=0.33, gamma:float=0.33, shift:float=1.) -> TYPE_IMAGE:
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

def image_stereo_shift(image: TYPE_IMAGE, depth: TYPE_IMAGE, shift:float=10) -> TYPE_IMAGE:
    # Ensure base image has alpha
    image = image_convert(image, 4)
    depth = image_convert(depth, 1)
    deltas = np.array((depth / 255.0) * float(shift), dtype=int)
    shifted_data = np.zeros(image.shape, dtype=np.uint8)
    _, width = image.shape[:2]
    for y, row in enumerate(deltas):
        for x, dx in enumerate(row):
            x2 = x + dx
            if (x2 >= width) or (x2 < 0):
                continue
            shifted_data[y][x2] = image[y][x]

    shifted_image = cv2pil(shifted_data)
    alphas_image = Image.fromarray(
        ndimage.binary_fill_holes(
            ImageChops.invert(
                shifted_image.getchannel("A")
            )
        )
    ).convert("1")
    shifted_image.putalpha(ImageChops.invert(alphas_image))
    return pil2cv(shifted_image)

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

def image_translate(image: TYPE_IMAGE, offset: TYPE_COORD = (0.0, 0.0), edge: EnumEdge = EnumEdge.CLIP, border_value:int=0) -> TYPE_IMAGE:
    """
    Translates an image by a given offset. Supports various edge handling methods.

    Args:
        image (TYPE_IMAGE): Input image as a numpy array.
        offset (TYPE_COORD): Tuple (offset_x, offset_y) representing the translation offset.
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

def image_transform(image: TYPE_IMAGE, offset:TYPE_COORD=(0.0, 0.0), angle:float=0, scale:TYPE_COORD=(1.0, 1.0), sample:EnumInterpolation=EnumInterpolation.LANCZOS4, edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:
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

# KERNELS

def MEDIAN3x3(image: TYPE_IMAGE) -> TYPE_IMAGE:
    height, width = image.shape[:2]
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
    array([[ 0, 1, 1],
        [-1, 0, 1],
        [-1, -1, 0]], dtype=int8)
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
# === COLOR FUNCTIONS ===
# =============================================================================

@cuda.jit
def kmeans_kernel(pixels, centroids, assignments) -> None:
    idx = cuda.grid(1)
    if idx < pixels.shape[0]:
        min_dist = 1e10
        min_centroid = 0
        for i in range(centroids.shape[0]):
            dist = 0
            for j in range(3):
                diff = pixels[idx, j] - centroids[i, j]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                min_centroid = i
        assignments[idx] = min_centroid

def color_image2lut(image: np.ndarray, num_colors: int = 256) -> np.ndarray:
    """Create X sized LUT from an RGB image using GPU acceleration."""
    # Ensure image is in RGB format
    if image.shape[2] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Reshape and transfer to GPU
    pixels = cp.asarray(image.reshape(-1, 3)).astype(cp.float32)
    # logger.debug("Pixel range:", cp.min(pixels), cp.max(pixels))

    # Initialize centroids using random pixels
    random_indices = cp.random.choice(pixels.shape[0], size=num_colors, replace=False)
    centroids = pixels[random_indices]
    # logger.debug("Initial centroids range:", cp.min(centroids), cp.max(centroids))

    # Prepare for K-means
    assignments = cp.zeros(pixels.shape[0], dtype=cp.int32)
    threads_per_block = 256
    blocks = (pixels.shape[0] + threads_per_block - 1) // threads_per_block

    # K-means iterations
    for iteration in range(20):  # Adjust the number of iterations as needed
        kmeans_kernel[blocks, threads_per_block](pixels, centroids, assignments)
        new_centroids = cp.zeros((num_colors, 3), dtype=cp.float32)
        for i in range(num_colors):
            mask = (assignments == i)
            if cp.any(mask):
                new_centroids[i] = cp.mean(pixels[mask], axis=0)

        centroids = new_centroids

        if iteration % 5 == 0:
            # logger.debug(f"Iteration {iteration}, Centroids range: {cp.min(centroids)} {cp.max(centroids)}")
            pass

    # Create LUT
    lut = cp.zeros((256, 1, 3), dtype=cp.uint8)
    lut[:num_colors] = cp.clip(centroids, 0, 255).reshape(-1, 1, 3).astype(cp.uint8)
    # logger.debug(f"Final LUT range: { cp.min(lut)} {cp.max(lut)}")
    return cp.asnumpy(lut)

def color_match_histogram(image: TYPE_IMAGE, usermap: TYPE_IMAGE) -> TYPE_IMAGE:
    """Colorize one input based on the histogram matches."""
    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 4:
        alpha = image_mask(image)
    image = image_convert(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    beta = cv2.cvtColor(usermap, cv2.COLOR_BGR2LAB)
    image = exposure.match_histograms(image, beta, channel_axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    image = image_blend(usermap, image, blendOp=BlendType.LUMINOSITY)
    image = image_convert(image, cc)
    if cc == 4:
        image[..., 3] = alpha[..., 0]
    return image

def color_match_reinhard(image: TYPE_IMAGE, target: TYPE_IMAGE) -> TYPE_IMAGE:
    """
    Apply Reinhard color matching to an image based on a target image.
    Works only for BGR images and returns an BGR image.

    based on https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.

    Args:
        image (TYPE_IMAGE): The input image (BGR or BGRA or Grayscale).
        target (TYPE_IMAGE): The target image (BGR or BGRA or Grayscale).

    Returns:
        TYPE_IMAGE: The color-matched image in BGR format.
    """
    target = image_convert(target, 3)
    lab_tar = cv2.cvtColor(target, cv2.COLOR_BGR2Lab)
    image = image_convert(image, 3)
    lab_ori = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    mean_tar, std_tar = cv2.meanStdDev(lab_tar)
    mean_ori, std_ori = cv2.meanStdDev(lab_ori)
    ratio = (std_tar / std_ori).reshape(-1)
    offset = (mean_tar - mean_ori * std_tar / std_ori).reshape(-1)
    lab_tar = cv2.convertScaleAbs(lab_ori * ratio + offset)
    return cv2.cvtColor(lab_tar, cv2.COLOR_Lab2BGR)

def color_match_lut(image: TYPE_IMAGE, colormap:int=cv2.COLORMAP_JET,
                    usermap:TYPE_IMAGE=None, num_colors:int=255) -> TYPE_IMAGE:
    """Colorize one input based on built in cv2 color maps or a user defined image."""
    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 4:
        alpha = image_mask(image)

    image = image_convert(image, 3)
    if usermap is not None:
        usermap = image_convert(usermap, 3)
        colormap = color_image2lut(usermap, num_colors)

    image = cv2.applyColorMap(image, colormap)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, 0.5, image, 0.5, 0)
    image = image_convert(image, cc)

    if cc == 4:
        image[..., 3] = alpha[..., 0]
    return image

def color_mean(image: TYPE_IMAGE) -> TYPE_IMAGE:
    color = [0, 0, 0]
    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 1:
        raw = int(np.mean(image))
        color = [raw] * 3
    else:
        # each channel....
        color = [
            int(np.mean(image[..., 0])),
            int(np.mean(image[:,:,1])),
            int(np.mean(image[:,:,2])) ]
    return color

def color_theory_complementary(color: TYPE_PIXEL) -> TYPE_PIXEL:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    return hsv2bgr(color_a)

def color_theory_monochromatic(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    sat = 255 / 5
    val = 255 / 5
    color_a = pixel_hsv_adjust(color, 0, -1 * sat, -1 * val, mod_sat=True, mod_value=True)
    color_b = pixel_hsv_adjust(color, 0, -2 * sat, -2 * val, mod_sat=True, mod_value=True)
    color_c = pixel_hsv_adjust(color, 0, -3 * sat, -3 * val, mod_sat=True, mod_value=True)
    color_d = pixel_hsv_adjust(color, 0, -4 * sat, -4 * val, mod_sat=True, mod_value=True)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c), hsv2bgr(color_d)

def color_theory_split_complementary(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 75, 0, 0)
    color_b = pixel_hsv_adjust(color, 105, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b)

def color_theory_analogous(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 30, 0, 0)
    color_b = pixel_hsv_adjust(color, 15, 0, 0)
    color_c = pixel_hsv_adjust(color, 165, 0, 0)
    color_d = pixel_hsv_adjust(color, 150, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c), hsv2bgr(color_d)

def color_theory_triadic(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 60, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b)

def color_theory_compound(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    color_c = pixel_hsv_adjust(color, 150, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c)

def color_theory_square(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 45, 0, 0)
    color_b = pixel_hsv_adjust(color, 90, 0, 0)
    color_c = pixel_hsv_adjust(color, 135, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c)

def color_theory_tetrad_custom(color: TYPE_PIXEL, delta:int=0) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)

    # modulus on neg and pos
    while delta < 0:
        delta += 90

    if delta > 90:
        delta = delta % 90

    color_a = pixel_hsv_adjust(color, -delta, 0, 0)
    color_b = pixel_hsv_adjust(color, delta, 0, 0)
    # just gimme a compliment
    color_c = pixel_hsv_adjust(color, 90 - delta, 0, 0)
    color_d = pixel_hsv_adjust(color, 90 + delta, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c), hsv2bgr(color_d)

def color_theory(image: TYPE_IMAGE, custom:int=0, scheme: EnumColorTheory=EnumColorTheory.COMPLIMENTARY) -> Tuple[TYPE_IMAGE, ...]:

    b = [0,0,0]
    c = [0,0,0]
    d = [0,0,0]
    color = color_mean(image)
    match scheme:
        case EnumColorTheory.COMPLIMENTARY:
            a = color_theory_complementary(color)
        case EnumColorTheory.MONOCHROMATIC:
            a, b, c, d = color_theory_monochromatic(color)
        case EnumColorTheory.SPLIT_COMPLIMENTARY:
            a, b = color_theory_split_complementary(color)
        case EnumColorTheory.ANALOGOUS:
            a, b, c, d = color_theory_analogous(color)
        case EnumColorTheory.TRIADIC:
            a, b = color_theory_triadic(color)
        case EnumColorTheory.SQUARE:
            a, b, c = color_theory_square(color)
        case EnumColorTheory.COMPOUND:
            a, b, c = color_theory_compound(color)
        case EnumColorTheory.CUSTOM_TETRAD:
            a, b, c, d = color_theory_tetrad_custom(color, custom)

    h, w = image.shape[:2]
    return (
        np.full((h, w, 3), color, dtype=np.uint8),
        np.full((h, w, 3), a, dtype=np.uint8),
        np.full((h, w, 3), b, dtype=np.uint8),
        np.full((h, w, 3), c, dtype=np.uint8),
        np.full((h, w, 3), d, dtype=np.uint8),
    )

# =============================================================================

def coord_cart2polar(x: float, y: float) -> Tuple[float, ...]:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def coord_polar2cart(r: float, theta: float) -> Tuple[float, ...]:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def coord_default(width:int, height:int, origin:Tuple[float, ...]=None) -> Tuple[float, ...]:
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    if origin is None:
        origin_x, origin_y = width // 2, height // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x -= origin_x
    y -= origin_y
    return x, y

def coord_fisheye(width: int, height: int, distortion: float) -> Tuple[TYPE_IMAGE, ...]:
    map_x, map_y = np.meshgrid(np.linspace(0., 1., width), np.linspace(0., 1., height))
    # normalized
    xnd, ynd = (2 * map_x - 1), (2 * map_y - 1)
    rd = np.sqrt(xnd**2 + ynd**2)
    # fish-eye distortion
    condition = (dist := 1 - distortion * (rd**2)) == 0
    xdu, ydu = np.where(condition, xnd, xnd / dist), np.where(condition, ynd, ynd / dist)
    xu, yu = ((xdu + 1) * width) / 2, ((ydu + 1) * height) / 2
    return xu.astype(np.float32), yu.astype(np.float32)

def coord_perspective(width: int, height: int, pts: List[TYPE_COORD]) -> TYPE_IMAGE:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0], pts[:, 1]])
    return cv2.getPerspectiveTransform(object_pts, pts)

def coord_sphere(width: int, height: int, radius: float) -> Tuple[TYPE_IMAGE, ...]:
    theta, phi = np.meshgrid(np.linspace(0, TAU, width), np.linspace(0, np.pi, height))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    # z = radius * np.cos(phi)
    x_image = (x + 1) * (width - 1) / 2
    y_image = (y + 1) * (height - 1) / 2
    return x_image.astype(np.float32), y_image.astype(np.float32)

def remap_fisheye(image: TYPE_IMAGE, distort: float) -> TYPE_IMAGE:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    map_x, map_y = coord_fisheye(width, height, distort)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #if cc == 1:
    #    image = image[..., 0]
    return image

def remap_perspective(image: TYPE_IMAGE, pts: list) -> TYPE_IMAGE:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pts = coord_perspective(width, height, pts)
    image = cv2.warpPerspective(image, pts, (width, height))
    #if cc == 1:
    #    image = image[..., 0]
    return image

def remap_polar(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Re-projects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    h, w = image.shape[:2]
    radius = max(w, h)
    return cv2.linearPolar(image, (h // 2, w // 2), radius // 2, cv2.WARP_INVERSE_MAP)

def remap_sphere(image: TYPE_IMAGE, radius: float) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    map_x, map_y = coord_sphere(width, height, radius)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def depth_from_gradient(grad_x, grad_y):
    """Optimized Frankot-Chellappa depth-from-gradient algorithm."""
    rows, cols = grad_x.shape
    rows_scale = np.fft.fftfreq(rows)
    cols_scale = np.fft.fftfreq(cols)
    u_grid, v_grid = np.meshgrid(cols_scale, rows_scale)
    grad_x_F = np.fft.fft2(grad_x)
    grad_y_F = np.fft.fft2(grad_y)
    denominator = u_grid**2 + v_grid**2
    denominator[0, 0] = 1.0
    Z_F = (-1j * u_grid * grad_x_F - 1j * v_grid * grad_y_F) / denominator
    Z_F[0, 0] = 0.0
    Z = np.fft.ifft2(Z_F).real
    Z -= np.min(Z)
    Z /= np.max(Z)
    return Z

def height_from_normal(image: TYPE_IMAGE, tile:bool=True) -> TYPE_IMAGE:
    """Computes a height map from the given normal map."""
    image = np.transpose(image, (2, 0, 1))
    flip_img = np.flip(image, axis=1)
    grad_x, grad_y = (flip_img[0] - 0.5) * 2, (flip_img[1] - 0.5) * 2
    grad_x = np.flip(grad_x, axis=0)
    grad_y = np.flip(grad_y, axis=0)

    if not tile:
        grad_x, grad_y = image_mirror_mandela(grad_x, grad_y)
    pred_img = depth_from_gradient(-grad_x, grad_y)

    # re-crop
    if not tile:
        height, width = image.shape[1], image.shape[2]
        pred_img = pred_img[:height, :width]

    image = np.stack([pred_img, pred_img, pred_img])
    image = np.transpose(image, (1, 2, 0))
    return image

def curvature_from_normal(image: TYPE_IMAGE, blur_radius:int=2)-> TYPE_IMAGE:
    """Computes a curvature map from the given normal map."""
    image = np.transpose(image, (2, 0, 1))
    blur_factor = 1 / 2 ** min(8, max(2, blur_radius))
    diff_kernel = np.array([-1, 0, 1])

    def conv_1d(array, kernel) -> np.ndarray[Any, np.dtype[Any]]:
        """Performs row-wise 1D convolutions with repeat padding."""
        k_l = len(kernel)
        extended = np.pad(array, k_l // 2, mode="wrap")
        return np.array([np.convolve(row, kernel, mode="valid") for row in extended[k_l//2:-k_l//2+1]])

    h_conv = conv_1d(image[0], diff_kernel)
    v_conv = conv_1d(-image[1].T, diff_kernel).T
    edges_conv = h_conv + v_conv

    # Calculate blur radius in pixels
    blur_radius_px = int(np.mean(image.shape[1:3]) * blur_factor)
    if blur_radius_px < 2:
        # If blur radius is too small, just normalize the edge convolution
        image = (edges_conv - np.min(edges_conv)) / (np.ptp(edges_conv) + 1e-10)
    else:
        blur_radius_px += blur_radius_px % 2 == 0

        # Compute Gaussian kernel
        sigma = max(1, blur_radius_px // 8)
        x = np.linspace(-(blur_radius_px - 1) / 2, (blur_radius_px - 1) / 2, blur_radius_px)
        g_kernel = np.exp(-0.5 * np.square(x) / np.square(sigma))
        g_kernel /= np.sum(g_kernel)

        # Apply Gaussian blur
        h_blur = conv_1d(edges_conv, g_kernel)
        v_blur = conv_1d(h_blur.T, g_kernel).T
        image = (v_blur - np.min(v_blur)) / (np.ptp(v_blur) + 1e-10)

    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.astype(np.uint8)

def roughness_from_normal(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Roughness from a normal map."""
    up_vector = np.array([0, 0, 1])
    image = 1 - np.dot(image, up_vector)
    image = (image - image.min()) / (image.max() - image.min())
    image = (255 * image).astype(np.uint8)
    return image_grayscale(image)

def roughness_from_albedo(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Roughness from an albedo map."""
    kernel_size = 3
    image = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    image = (image - image.min()) / (image.max() - image.min())
    image = (255 * image).astype(np.uint8)
    return image_grayscale(image)

def roughness_from_albedo_normal(albedo: TYPE_IMAGE, normal: TYPE_IMAGE, blur:int=2, blend:float=0.5, iterations:int=3) -> TYPE_IMAGE:
    normal = roughness_from_normal(normal)
    normal = image_normalize(normal)
    albedo = roughness_from_albedo(albedo)
    albedo = image_normalize(albedo)
    rough = image_lerp(normal, albedo, alpha=blend)
    rough = image_normalize(rough)
    image = image_lerp(normal, rough, alpha=blend)
    iterations = min(16, max(2, iterations))
    blur += (blur % 2 == 0)
    step = 1 / 2 ** iterations
    for i in range(iterations):
        image = cv2.add(normal * step, image * step)
        image = cv2.GaussianBlur(image, (blur + i * 2, blur + i * 2), 3 * i)

    inverted = 255 - image_normalize(image)
    inverted = cv2.subtract(inverted, albedo) * 0.5
    inverted = cv2.GaussianBlur(inverted, (blur, blur), blur)
    inverted = image_normalize(inverted)

    image = cv2.add(image * 0.5, inverted * 0.5)
    for i in range(iterations):
        image = cv2.GaussianBlur(image, (blur, blur), blur)

    image = cv2.add(image * 0.5, inverted * 0.5)
    for i in range(iterations):
        image = cv2.GaussianBlur(image, (blur, blur), blur)

    image = image_normalize(image)
    return image
