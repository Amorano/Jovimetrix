"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Image Support
"""

import math
import base64
import urllib
import requests
from enum import Enum
from io import BytesIO
from typing import Any, Optional, Tuple, Union

import cv2
import torch
import numpy as np
from numba import jit
from daltonlens import simulate
from sklearn.cluster import MiniBatchKMeans

from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageOps
from blendmodes.blend import blendLayers, BlendType

from loguru import logger

from Jovimetrix.sup.util import grid_make

# =============================================================================
# === GLOBAL ===
# =============================================================================

MIN_IMAGE_SIZE = 512
HALFPI = math.pi / 2
TAU = math.pi * 2

# =============================================================================
# === TYPE SHORTCUTS ===
# =============================================================================

TYPE_COORD = Union[
    tuple[int, int],
    tuple[float, float]
]

TYPE_PIXEL = Union[
    int,
    float,
    Tuple[float, float, float],
    Tuple[float, float, float, Optional[float]],
    Tuple[int, int, int],
    Tuple[int, int, int, Optional[int]]
]

TYPE_IMAGE = Union[np.ndarray, torch.Tensor]
TYPE_VECTOR = Union[TYPE_IMAGE|TYPE_PIXEL]

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
    NONE = 0
    CROP = 20
    MATTE = 25
    FIT = 10
    ASPECT = 30
    ASPECT_SHORT = 35

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

class EnumSwizzle(Enum):
    A_X = 10
    A_Y = 11
    A_Z = 12
    A_W = 13
    B_X = 20
    B_Y = 21
    B_Z = 22
    B_W = 23
    CONSTANT = 40

class EnumCBSimulator(Enum):
    AUTOSELECT = 0
    BRETTEL1997 = 1
    COBLISV1 = 2
    COBLISV2 = 3
    MACHADO2009 = 4
    VIENOT1999 = 5
    VISCHECK = 6

class EnumCBDefiency(Enum):
    DEUTAN = simulate.Deficiency.DEUTAN
    PROTAN = simulate.Deficiency.PROTAN
    TRITAN = simulate.Deficiency.TRITAN

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
    Formula taken from Wikipedia https://en.wikipedia.org/wiki/SRGB
    """
    image = image.astype(float) / 255.
    gamma = ((image + 0.055) / 1.055) ** 2.4
    scale = image / 12.92
    image = np.where (image > 0.04045, gamma, scale)
    return (image * 255).astype(np.uint8)

def linear2sRGB(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Convert linearRGB to sRGB, applying the gamma correction.
    Formula taken from Wikipedia https://en.wikipedia.org/wiki/SRGB
    """
    image = image.astype(float) / 255.
    cutoff = image < 0.0031308
    higher = 1.055 * pow(image, 1.0 / 2.4) - 0.055
    lower = image * 12.92
    image = np.where (image > cutoff, higher, lower)
    return (image * 255).astype(np.uint8)

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

def cv2pil(image: TYPE_IMAGE, chan:EnumImageType=EnumImageType.RGBA) -> Image.Image:
    if image is None:
        return channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 0, chan)
    mode = cv2.COLOR_BGR2GRAY
    if chan == EnumImageType.RGB:
        mode = cv2.COLOR_BGR2RGBA
    elif chan == EnumImageType.RGBA:
        mode = cv2.COLOR_BGRA2RGBA
    image = cv2.cvtColor(image, mode)
    return Image.fromarray(image)

def cv2tensor(image: TYPE_IMAGE) -> torch.Tensor:
    """Convert a CV2 Matrix to a Torch Tensor."""
    cc = 1 if len(image.shape) < 3 else image.shape[2]
    match cc:
        case 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        case 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        case 1:
            if len(image.shape) > 2:
                image = image.squeeze()
    return torch.from_numpy(image.astype(np.float32) / 255).unsqueeze(0)

def cv2tensor_full(image: TYPE_IMAGE, matte:TYPE_PIXEL=0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = image_mask(image)
    #if channel_count(image)[0] != 4:
    #    image = image_convert(image, 4)
    image = image_matte(image, matte)
    rgb = image_convert(image, 3)
    return cv2tensor(image), cv2tensor(rgb), cv2tensor(mask)

def hsv2bgr(hsl_color: TYPE_PIXEL) -> TYPE_PIXEL:
    return cv2.cvtColor(np.uint8([[hsl_color]]), cv2.COLOR_HSV2BGR)[0, 0]

def image2bgr(image: TYPE_IMAGE) -> tuple[int, TYPE_IMAGE, TYPE_IMAGE]:
    """RGB Helper function.
    Return channel count, BGR, and Alpha.
    """
    alpha = image_mask(image)
    if (cc := channel_count(image)[0]) == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif (cc := channel_count(image)[0]) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image, alpha, cc

def pil2cv(image: Image.Image, chan:EnumImageType=EnumImageType.BGRA) -> TYPE_IMAGE:
    """Convert a PIL Image to a CV2 Matrix."""
    mode = image.mode
    image = np.array(image).astype(np.uint8)
    if chan == EnumImageType.BGRA:
        if mode == 'RGBA':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif mode == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        else:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif chan == EnumImageType.BGR:
        if mode == 'RGBA':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif mode == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif chan == EnumImageType.GRAYSCALE:
        if mode == 'RGBA':
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif mode == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a Torch Tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255).unsqueeze(0)

def tensor2cv(tensor: torch.Tensor, chan:EnumImageType=EnumImageType.BGRA, width:int=MIN_IMAGE_SIZE, height:int=MIN_IMAGE_SIZE, matte:TYPE_PIXEL=(0, 0, 0, 255)) -> TYPE_IMAGE:
    if not isinstance(tensor, (torch.Tensor,)):
        return channel_solid(width, height, matte, chan=chan)
    image = np.clip(tensor.squeeze().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    cc = 1 if len(image.shape) < 3 else image.shape[2]
    if chan == EnumImageType.BGRA:
        if cc == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif cc == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif chan == EnumImageType.RGBA:
        if cc == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        elif cc == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif chan == EnumImageType.BGR:
        if cc == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif cc == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif chan == EnumImageType.RGB:
        if cc == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif cc == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif chan == EnumImageType.GRAYSCALE:
        if cc == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif cc == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.expand_dims(image, -1)

def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a torch Tensor to a PIL Image."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    cc = 1 if len(tensor.shape) < 3 else tensor.shape[2]
    if cc == 4:
        return Image.fromarray(tensor, mode='RGBA')
    elif cc == 3:
        return Image.fromarray(tensor, mode='RGB')
    return Image.fromarray(tensor, mode='L')

# =============================================================================
# === PIXEL ===
# =============================================================================

def pixel_eval(color: TYPE_PIXEL,
            target: EnumImageType=EnumImageType.BGR,
            precision:EnumIntFloat=EnumIntFloat.INT,
            crunch:EnumGrayscaleCrunch=EnumGrayscaleCrunch.MEAN) -> tuple[TYPE_PIXEL] | TYPE_PIXEL:

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

def pixel_hsv_adjust(color:TYPE_PIXEL, hue:int=0, saturation:int=0, value:int=0,
            mod_color:bool=True, mod_sat:bool=False, mod_value:bool=False) -> TYPE_PIXEL:
    """Adjust an HSV type pixel.
    OpenCV uses... H: 0-179, S: 0-255, V: 0-255"""
    hsv = [0, 0, 0]
    hsv[0] = (color[0] + hue) % 180 if mod_color else np.clip(color[0] + hue, 0, 180)
    hsv[1] = (color[1] + saturation) % 255 if mod_sat else np.clip(color[1] + saturation, 0, 255)
    hsv[2] = (color[2] + value) % 255 if mod_value else np.clip(color[2] + value, 0, 255)
    return hsv

def pixel_convert(color:TYPE_PIXEL, size:int=4, alpha:int=255) -> TYPE_PIXEL:
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

def channel_count(image:TYPE_IMAGE) -> tuple[int, int, int, EnumImageType]:
    size = image.shape[2] if len(image.shape) > 2 else 1
    if len(image.shape) > 1:
        h, w = image.shape[:2]
    else:
        h = w = image.shape[0]
    if size == 4:
        mode = EnumImageType.BGRA
        if type(image) == Image:
            mode = EnumImageType.RGBA
    elif size == 3:
        mode = EnumImageType.BGR
        if type(image) == Image:
            mode = EnumImageType.RGB
    else:
        mode = EnumImageType.GRAYSCALE
    return size, w, h, mode

def channel_add(image:TYPE_IMAGE, color:TYPE_PIXEL=255) -> TYPE_IMAGE:
    h, w = image.shape[:2]
    color = pixel_eval(color, EnumImageType.GRAYSCALE)
    new = channel_solid(w, h, color, EnumImageType.GRAYSCALE)
    return np.concatenate([image, new], axis=-1)

def channel_solid(width:int, height:int, color:TYPE_PIXEL=(0, 0, 0, 0),
                  chan:EnumImageType=EnumImageType.BGR) -> TYPE_IMAGE:

    if chan == EnumImageType.GRAYSCALE:
        color = pixel_eval(color, EnumImageType.GRAYSCALE)
        return np.full((height, width, 1), color, dtype=np.uint8)

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

def channel_merge(channel:list[TYPE_IMAGE]) -> TYPE_IMAGE:
    ch = [c.shape[:2] if c is not None else (0, 0) for c in channel[:3]]
    w = max([c[1] for c in ch])
    h = max([c[0] for c in ch])
    ch = [np.zeros((h, w), dtype=np.uint8) if c is None else c for c in channel[:3]]
    if len(channel) == 4:
        a = channel[3] if len(channel) == 4 else np.full((h, w), 255, dtype=np.uint8)
        ch.append(a)
    return cv2.merge(ch)

def channel_swap(imageA:TYPE_IMAGE, swap_ot:EnumPixelSwizzle,
                 imageB:TYPE_IMAGE, swap_in:EnumPixelSwizzle) -> TYPE_IMAGE:
    index_out = int(swap_ot.value / 10)
    cc_out = channel_count(imageA)[0]
    # swap channel is out of range of image size
    if index_out > cc_out:
        return imageA
    index_in = int(swap_in.value / 10)
    cc_in = channel_count(imageB)[0]
    if index_in > cc_in:
        return imageA
    h, w = imageA.shape[:2]

    # imageB = image_crop_center(imageB, w, h)
    # imageB = image_matte(imageB, width=w, height=h)
    imageB = image_scalefit(imageB, w, h, EnumScaleMode.FIT)
    imageA[:,:,index_out] = imageB[:,:,index_in]
    return imageA

# =============================================================================
# === EXPLICIT SHAPE FUNCTIONS ===
# =============================================================================

def shape_body(func: str, width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    back = pixel_eval(back, EnumImageType.RGB)
    image = Image.new("RGB", (width, height), back)
    d = ImageDraw.Draw(image)
    func = getattr(d, func)
    fill = pixel_eval(fill, EnumImageType.RGBA)
    func(xy, fill=fill)
    return image

def shape_ellipse(width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    return shape_body('ellipse', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill, back=back)

def shape_quad(width: int, height: int, sizeX:float=1., sizeY:float=1., fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    return shape_body('rectangle', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill, back=back)

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
    imageB = image_crop_center(imageB, w, h)
    imageB = image_matte(imageB, (0,0,0,0), w, h)
    old_mask = image_mask(imageB)[:,:,0]
    if mask is not None:
        mask = image_crop_center(mask, w, h)
        mask = image_matte(mask, (0,0,0,0), w, h)
        mask = image_convert(mask, 1)
        old_mask = cv2.bitwise_and(mask, old_mask)
    imageB[:,:,3] = old_mask
    imageB = cv2pil(imageB)
    image = blendLayers(imageA, imageB, blendOp.value, np.clip(alpha, 0, 1))
    image = pil2cv(image)
    return image_crop_center(image, w, h)

def image_color_blind(image: TYPE_IMAGE, deficiency:EnumCBDefiency,
                      simulator:EnumCBSimulator=EnumCBSimulator.AUTOSELECT,
                      severity:float=1.0) -> TYPE_IMAGE:

    if (cc := channel_count(image)[0]) == 4:
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
    """Force image format to number of channels chosen."""
    ncc = max(1, min(4, channels))
    if ncc < 3:
        return image_grayscale(image)
    cc = channel_count(image)[0]
    if ncc == cc:
        return image
    elif ncc == 3:
        if cc == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if cc == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

def image_crop_polygonal(image: TYPE_IMAGE, points: list[TYPE_COORD]) -> TYPE_IMAGE:
    cc, w, h = channel_count(image)[:3]
    mask = image_mask(image, 0)
    # crop area first
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    point_mask = np.zeros((h, w, 1), dtype=np.uint8)
    point_mask = cv2.fillPoly(point_mask, [points], 255)
    x, y, w, h = cv2.boundingRect(point_mask)
    point_mask = point_mask[:,:,0]
    # store any alpha channel
    if cc == 4:
        mask = image_mask(image, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # crop with the point_mask
    image = cv2.bitwise_and(image, image, mask=point_mask)
    image = image[y:y+h, x:x+w, :3]
    # replace old alpha channel
    if cc == 4:
        return image_mask_add(image, mask[y:y+h, x:x+w])
    elif cc == 1:
        return image_convert(image, cc)
    return image

def image_crop(image: TYPE_IMAGE, width:int=None, height:int=None, offset:tuple[float, float]=(0, 0)) -> TYPE_IMAGE:
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
    width = width if width is not None else w
    height = height if height is not None else h
    y = max(0, int((h - height) / 2))
    x = max(0, int((w - width)/ 2))
    points = [(x, y), (x + width - 1, y), (x + width - 1, y + height - 1), (x, y + height - 1)]
    return image_crop_polygonal(image, points)

def image_diff(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, threshold:int=0, color:TYPE_PIXEL=(255, 0, 0)) -> tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, float]:
    _, w1, h1 = channel_count(imageA)[:3]
    _, w2, h2 = channel_count(imageB)[:3]
    w1 = max(w1, w2)
    h1 = max(h1, h2)
    imageA = image_matte(imageA, (0, 0, 0, 0), w1, h1)
    imageA = image_convert(imageA, 3)
    imageB = image_matte(imageB, (0, 0, 0, 0), w1, h1)
    imageB = image_convert(imageB, 3)
    grayA = image_grayscale(imageA)
    grayB = image_grayscale(imageB)
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # h, w = imageA.shape[:2]
    high_a = imageA.copy()
    high_a = image_convert(high_a, 3)
    # h, w = imageB.shape[:2]
    # high_b = np.zeros((h, w, 3), dtype=np.uint8)
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

def image_filter(image:TYPE_IMAGE, matrix:list[float|int]) -> TYPE_IMAGE:
    """Apply a scalar matrix of numbers to each channel of an image.

    The matrix should be formed such that all the R scalars are first, G then B.
    """
    if channel_count(image)[0] < 3:
        image = image_convert(image, 3)
    image = image.astype(float)
    r = matrix[:3]
    g = matrix[3:6]
    b = matrix[6:]
    image[:,:,2] = (image[:,:,2] * r[0] + image[:,:,1] * r[1] + image[:,:,0] * r[2])
    image[:,:,1] = (image[:,:,2] * g[0] + image[:,:,1] * g[1] + image[:,:,0] * g[2])
    image[:,:,0] = (image[:,:,2] * b[0] + image[:,:,1] * b[1] + image[:,:,0] * b[2])
    return image.astype(np.uint8)

def image_formats() -> list[str]:
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

    def pixel(x, spread:int=1) -> tuple[int, int, int]:
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

def image_grayscale(image: TYPE_IMAGE) -> TYPE_IMAGE:
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    cc = channel_count(image)[0]
    if cc == 1:
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        return image
    if cc == 4:
        image = image[:,:,:3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2]

def image_grid(data: list[TYPE_IMAGE], width: int, height: int) -> TYPE_IMAGE:
    #@TODO: makes poor assumption all images are the same dimensions.
    chunks, col, row = grid_make(data)
    frame = np.zeros((height * row, width * col, 4), dtype=np.uint8)
    i = 0
    for y, strip in enumerate(chunks):
        for x, item in enumerate(strip):
            if channel_count(item)[0] == 3:
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
    value = np.clip(value, 0, 1)
    image, alpha, cc = image2bgr(image)
    image = cv2.addWeighted(image, 1 - value, 255 - image, value, 0)
    return bgr2image(image, alpha, cc == 1)

def image_lerp(imageA:TYPE_IMAGE,
              imageB:TYPE_IMAGE,
              mask:TYPE_IMAGE=None,
              alpha:float=1.) -> TYPE_IMAGE:

    imageA = imageA.astype(np.float32)
    imageB = imageB.astype(np.float32)

    # normalize alpha and establish mask
    alpha = np.clip(alpha, 0, 1)
    if mask is None:
        height, width = imageA.shape[:2]
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

def image_levels(image:torch.Tensor, black_point:int=0, white_point=255,
        mid_point=128, gamma=1.0) -> TYPE_IMAGE:

    image, alpha, cc = image2bgr(image)
    black  = np.array([black_point] * 3, dtype=np.float32)
    white  = np.array([white_point] * 3, dtype=np.float32)
    mid  = np.array([mid_point] * 3, dtype=np.float32)
    inGamma  = np.array([gamma] * 3, dtype=np.float32)
    outBlack = np.array([0, 0, 0], dtype=np.float32)
    outWhite = np.array([255, 255, 255], dtype=np.float32)
    image = np.clip( (image - black) / (white - black), 0, 255 )
    image = (image ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
    image = np.clip(image, 0, 255).astype(np.uint8)
    return bgr2image(image, alpha, cc == 1)

def image_load(url: str) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    """
    if img.format == 'PSD':
        images = [pil2cv(frame.copy()) for frame in ImageSequence.Iterator(img)]
        logger.debug(f"#PSD {len(images)}")
    """
    try:
        img = cv2.imread(url, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        try:
            img = Image.open(url)
            img = ImageOps.exif_transpose(img)
            img = pil2cv(img)
        except Exception as e:
            logger.error(str(e))
            img = np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8)
            mask = np.full((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), 255, dtype=np.uint8)
            return img, mask

    if img is None:
        raise Exception(f"no file {url}")
    if img.dtype != np.uint8:
        img = np.array(img * 255, dtype=np.uint8)
    return img, image_mask(img)

def image_load_data(data: str) -> TYPE_IMAGE:
    img = ImageOps.exif_transpose(data)
    return pil2cv(img)

def image_load_exr(url: str) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
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

def image_load_from_url(url:str) -> TYPE_IMAGE:
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
    """Returns a mask from an image or a default mask with the color."""
    cc, width, height = channel_count(image)[:3]
    if cc == 4:
        return np.expand_dims(image[:,:,3], -1)
    return channel_solid(width, height, color, EnumImageType.GRAYSCALE)

def image_mask_add(image:TYPE_IMAGE, mask:TYPE_IMAGE=None) -> TYPE_IMAGE:
    """Places a default or custom mask into an image.
    Images are expanded to 4 channels.
    Existing 4 channel images with no mask input just return themselves.
    """
    h, w = image.shape[:2]
    image = image_convert(image, 4)
    if mask is None:
        mask = image_mask(image)
    else:
        mask = image_grayscale(mask)
        mask = image_scalefit(mask, w, h, EnumScaleMode.FIT)
    image[:,:,3] = mask[:,:,0]
    return image

def image_matte(image:TYPE_IMAGE, color:TYPE_PIXEL=(0,0,0,255),
                width:int=None, height:int=None, imageB:TYPE_IMAGE=None) -> TYPE_IMAGE:
    """Puts an image atop a colored matte."""
    cc, w, h = channel_count(image)[:3]
    width = width if width is not None else w
    height = height if height is not None else h
    width = max(w, width)
    height = max(h, height)
    y1 = max(0, (height - h) // 2)
    y2 = min(height, y1 + h)
    x1 = max(0, (width - w) // 2)
    x2 = min(width, x1 + w)
    if cc != 4:
        image = image_convert(image, 4)
    # save the old alpha channel
    mask = image_mask(image)
    if imageB is not None:
        matte = image_convert(imageB, 4)
        matte = image_scalefit(matte, width, height, EnumScaleMode.FIT)
    else:
        matte = channel_solid(width, height, color, EnumImageType.BGRA)
    alpha = mask[:,:,0]
    matte[y1:y2, x1:x2, 3] = alpha
    alpha = cv2.bitwise_not(alpha)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGRA) / 255.0
    matte[y1:y2, x1:x2] = cv2.convertScaleAbs(image * (1 - alpha) + matte[y1:y2, x1:x2] * alpha)
    if cc == 4:
        matte[y1:y2, x1:x2,3] = mask[:,:,0]
    return matte

def image_merge(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, axis: int=0, flip: bool=False) -> TYPE_IMAGE:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def image_mirror(image: TYPE_IMAGE, mode:EnumMirrorMode, x:float=0.5, y:float=0.5) -> TYPE_IMAGE:
    cc, width, height = channel_count(image)[:3]

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

def image_quantize(image:TYPE_IMAGE, levels:int=256, iterations:int=10, epsilon:float=0.2) -> TYPE_IMAGE:
    levels = int(max(2, min(256, levels)))
    pixels = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    _, labels, centers = cv2.kmeans(pixels, levels, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(image.shape)

def image_rotate(image: TYPE_IMAGE, angle: float, center:TYPE_COORD=(0.5, 0.5), edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    def func_rotate(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        c = (int(width * center[0]), int(height * center[1]))
        M = cv2.getRotationMatrix2D(c, -angle, 1.0)
        return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR)

    image = image_affine_edge(image, func_rotate, edge)
    return image

def image_save_gif(fpath:str, images: list[Image.Image], fps: int=0,
                loop:int=0, optimize:bool=False) -> None:

    fps = min(50, max(1, fps))
    images[0].save(
        fpath,
        append_images=images[1:],
        duration=3,  # int(100.0 / fps),
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
                 mode:EnumScaleMode=EnumScaleMode.NONE,
                 sample:EnumInterpolation=EnumInterpolation.LANCZOS4,
                 matte:TYPE_PIXEL=(0,0,0,255)) -> TYPE_IMAGE:

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
            image = image_crop_center(image, width, height)

        case EnumScaleMode.FIT:
            image = cv2.resize(image, (width, height), interpolation=sample.value)

    if len(image.shape) == 2:
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

def image_split(image: TYPE_IMAGE) -> tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE]:
    cc, w, h = channel_count(image)[:3]
    if cc == 4:
        b, g, r, a = cv2.split(image)
    elif cc == 3:
        b, g, r = cv2.split(image)
        a = np.full((h, w), 255, dtype=np.uint8)
    else:
        r = g = b = image
        a = np.full((h, w), 255, dtype=np.uint8)
    return r, g, b, a

def image_stack(images: list[TYPE_IMAGE], axis:EnumOrientation=EnumOrientation.HORIZONTAL,
                stride:Optional[int]=None, matte:TYPE_PIXEL=(0,0,0,255)) -> TYPE_IMAGE:

    stack = []
    width, height = 0, 0
    for i in images:
        h, w = i.shape[:2]
        width = max(width, w)
        height = max(height, h)
        stack.append(i)

    images = [image_matte(image_convert(i, 4), matte, width, height) for i in stack]
    count = len(images)
    matte = pixel_convert(matte, 4)
    match axis:
        case EnumOrientation.GRID:
            if stride < 1:
                stride = np.ceil(np.sqrt(count))
                stride = int(stride)
            stride = min(stride, count)

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

def image_translate(image: TYPE_IMAGE, offset:TYPE_COORD=(0.0, 0.0), edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    def translate(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        scalarX = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPX] else 1.
        scalarY = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPY] else 1.

        M = np.float32([[1, 0, offset[0] * width * scalarX], [0, 1, offset[1] * height * scalarY]])
        return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR)

    return image_affine_edge(image, translate, edge)

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
        [-kernel,    -kernel+1,     0],
        [-kernel+1,    kernel-1,      1],
        [kernel-2,     kernel-1,      2]
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
# === COLOR FUNCTIONS ===
# =============================================================================

def color_image2lut(image: TYPE_IMAGE, num_colors:int=256) -> np.ndarray[np.uint8]:
    """Create X sized LUT from an RGB image."""
    image = image_convert(image, 3)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3)
    kmeans = MiniBatchKMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(np.uint8)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        lut[i] = color
    return lut

def color_match_histogram(image: TYPE_IMAGE, usermap: TYPE_IMAGE) -> TYPE_IMAGE:
    """Colorize one input based on the histogram matches."""
    if (cc := channel_count(image)[0]) == 4:
        alpha = image_mask(image)[:,:,0]
    image = image_convert(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    beta = cv2.cvtColor(usermap, cv2.COLOR_BGR2LAB)
    image = exposure.match_histograms(image, beta, channel_axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    image = image_blend(usermap, image, blendOp=BlendType.LUMINOSITY)
    image = image_convert(image, cc)
    if cc == 4:
        image[:,:,3] = alpha
    return image

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

def color_match_lut(image: TYPE_IMAGE, colormap:int=cv2.COLORMAP_JET,
                      usermap:TYPE_IMAGE=None, num_colors:int=255) -> TYPE_IMAGE:
    """Colorize one input based on built in cv2 color maps or a user defined image."""
    if (cc := channel_count(image)[0]) == 4:
        alpha = image_mask(image)[:,:,0]
    image = image_convert(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    if usermap is not None:
        usermap = image_convert(usermap, 3)
    colormap = colormap if usermap is None else color_image2lut(usermap, num_colors)
    image = cv2.applyColorMap(image, colormap)
    image = cv2.addWeighted(image, 0.5, image, 0.5, 0)
    image = image_convert(image, cc)
    if cc == 4:
        image[:,:,3] = alpha
    return image

def color_mean(image: TYPE_IMAGE) -> TYPE_IMAGE:
    color = [0, 0, 0]
    if channel_count(image)[0] == 1:
        raw = int(np.mean(image))
        color = [raw] * 3
    else:
        # each channel....
        color = [
            int(np.mean(image[:,:,0])),
            int(np.mean(image[:,:,1])),
            int(np.mean(image[:,:,2])) ]
    return color

def color_theory_complementary(color: TYPE_PIXEL) -> TYPE_PIXEL:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    return hsv2bgr(color_a)

def color_theory_monochromatic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = bgr2hsv(color)
    sat = 255 / 5
    val = 255 / 5
    color_a = pixel_hsv_adjust(color, 0, -1 * sat, -1 * val, mod_sat=True, mod_value=True)
    color_b = pixel_hsv_adjust(color, 0, -2 * sat, -2 * val, mod_sat=True, mod_value=True)
    color_c = pixel_hsv_adjust(color, 0, -3 * sat, -3 * val, mod_sat=True, mod_value=True)
    color_d = pixel_hsv_adjust(color, 0, -4 * sat, -4 * val, mod_sat=True, mod_value=True)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c), hsv2bgr(color_d)

def color_theory_split_complementary(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 75, 0, 0)
    color_b = pixel_hsv_adjust(color, 105, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b)

def color_theory_analogous(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 30, 0, 0)
    color_b = pixel_hsv_adjust(color, 15, 0, 0)
    color_c = pixel_hsv_adjust(color, 165, 0, 0)
    color_d = pixel_hsv_adjust(color, 150, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c), hsv2bgr(color_d)

def color_theory_triadic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 60, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b)

def color_theory_compound(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    color_c = pixel_hsv_adjust(color, 150, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c)

def color_theory_square(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 45, 0, 0)
    color_b = pixel_hsv_adjust(color, 90, 0, 0)
    color_c = pixel_hsv_adjust(color, 135, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c)

def color_theory_tetrad_custom(color: TYPE_PIXEL, delta:int=0) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
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

def color_theory(image: TYPE_IMAGE, custom:int=0, scheme: EnumColorTheory=EnumColorTheory.COMPLIMENTARY) -> tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE]:

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

def coord_cart2polar(x, y) -> tuple[Any, Any]:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def coord_polar2cart(r, theta) -> tuple:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def coord_default(width:int, height:int, origin:tuple[float, float]=None) -> tuple:
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

def coord_perspective(width: int, height: int, pts: list[TYPE_COORD]) -> TYPE_IMAGE:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0], pts[:, 1]])
    return cv2.getPerspectiveTransform(object_pts, pts)

def coord_sphere(width: int, height: int, radius: float) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    theta, phi = np.meshgrid(np.linspace(0, TAU, width), np.linspace(0, np.pi, height))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    # z = radius * np.cos(phi)
    x_image = (x + 1) * (width - 1) / 2
    y_image = (y + 1) * (height - 1) / 2
    return x_image.astype(np.float32), y_image.astype(np.float32)

def remap_fisheye(image: TYPE_IMAGE, distort: float) -> TYPE_IMAGE:
    cc, width, height = channel_count(image)[:3]
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    map_x, map_y = coord_fisheye(width, height, distort)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if cc == 1:
        image = image[:,:,0][:,:]
    return image

def remap_perspective(image: TYPE_IMAGE, pts: list) -> TYPE_IMAGE:
    cc, width, height = channel_count(image)[:3]
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pts = coord_perspective(width, height, pts)
    image = cv2.warpPerspective(image, pts, (width, height))
    if cc == 1:
        image = image[:,:,0][:,:]
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
