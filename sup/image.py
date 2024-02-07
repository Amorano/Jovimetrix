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
from typing import Any, Optional

import cv2
import torch
import numpy as np
import scipy as sp

from PIL import Image, ImageDraw, ImageOps, ImageSequence
from blendmodes.blend import blendLayers, BlendType
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from loguru import logger

from Jovimetrix import TYPE_IMAGE, TYPE_PIXEL, TYPE_COORD, IT_WH, MIN_IMAGE_SIZE
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import grid_make, deep_merge_dict

# =============================================================================
# === ENUM GLOBALS ===
# =============================================================================

HALFPI = math.pi / 2
TAU = math.pi * 2

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
    TWIimage_SHIFTED = cv2.COLORMAP_TWILIGHT_SHIFTED
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
    RGB = 1
    RGBA = 2

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

class EnumScaleMode(Enum):
    NONE = 0
    FIT = 1
    CROP = 2
    ASPECT = 3

class EnumThreshold(Enum):
    BINARY = cv2.THRESH_BINARY
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO

class EnumThresholdAdapt(Enum):
    ADAPT_NONE = -1
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADAPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# =============================================================================
# === NODE SUPPORT ===
# =============================================================================

IT_SAMPLE = {"optional": {
    Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
}}

IT_SCALEMODE = {"optional": {
    Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
}}

IT_EDGE = {"optional": {
    Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
}}

IT_WHMODE = deep_merge_dict(IT_WH, IT_SCALEMODE)

# =============================================================================
# === MATRIX SUPPORT ===
# =============================================================================

def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a torch Tensor to a PIL Image."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    if len(tensor.shape) == 2:
        return Image.fromarray(tensor, mode='L')
    elif len(tensor.shape) == 3 and tensor.shape[2] == 3:
        return Image.fromarray(tensor, mode='RGB')
    elif len(tensor.shape) == 3 and tensor.shape[2] == 4:
        return Image.fromarray(tensor, mode='RGBA')

def tensor2cv(tensor: torch.Tensor) -> TYPE_IMAGE:
    """Convert a torch Tensor to a CV2 Matrix."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    if len(tensor.shape) == 2:
        return cv2.cvtColor(tensor, cv2.COLOR_GRAY2BGR)
    elif len(tensor.shape) == 3 and tensor.shape[2] == 3:
        return cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    elif len(tensor.shape) == 3 and tensor.shape[2] == 4:
        return cv2.cvtColor(tensor, cv2.COLOR_RGBA2BGRA)

def tensor2mask(tensor: torch.Tensor) -> TYPE_IMAGE:
    """Convert a torch Tensor to a Mask as a CV2 Matrix."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return tensor

def tensor2np(tensor: torch.Tensor) -> TYPE_IMAGE:
    """Convert a torch Tensor to a Numpy Array."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return tensor

def b64_2_tensor(base64str: str) -> torch.Tensor:
    img = base64.b64decode(base64str)
    img = Image.open(BytesIO(img))
    img = ImageOps.exif_transpose(img)
    return pil2tensor(img)

def mask2cv(mask: torch.Tensor) -> TYPE_IMAGE:
    """Convert a torch Tensor (Mask) to a CV2 Matrix."""
    tensor = np.clip(255 * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return cv2.cvtColor(tensor, cv2.COLOR_GRAY2BGR)

def mask2pil(mask: torch.Tensor) -> Image.Image:
    """Convert a torch Tensor (Mask) to a PIL Image."""
    tensor = np.clip(255 * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(tensor, mode='L')

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a Torch Tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2cv(image: Image.Image) -> TYPE_IMAGE:
    """Convert a PIL Image to a CV2 Matrix."""
    mode = cv2.COLOR_RGB2BGR if image.mode == 'RGBA' else cv2.COLOR_RGBA2BGRA
    return cv2.cvtColor(np.array(image), mode).astype(np.uint8)

def pil2mask(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a Torch Tensor (Mask)."""
    return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)

def cv2tensor(image: TYPE_IMAGE) -> torch.Tensor:
    """Convert a CV2 Matrix to a Torch Tensor."""
    if len(image.shape) == 2:
        # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # Grayscale image with an extra channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    elif len(image.shape) > 2 and image.shape[2] > 3:
        # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA).astype(np.float32)
    else:
        # RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(image / 255.0).unsqueeze(0)

def cv2mask(image: TYPE_IMAGE) -> torch.Tensor:
    """Convert a CV2 Matrix to a Torch Tensor (Mask)."""
    if len(image.shape) == 2:
        # Grayscale image
        return torch.from_numpy(image / 255.0).unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # Grayscale image with an extra channel
        return torch.from_numpy(image / 255.0).unsqueeze(0)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return torch.from_numpy(gray_image / 255.0).unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # RGBA image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
        return torch.from_numpy(gray_image / 255.0).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Unsupported image format")

def cv2pil(image: TYPE_IMAGE) -> Image.Image:
    """Convert a CV2 Matrix to a PIL Image."""
    if len(image.shape) == 2:
        # Grayscale image
        return Image.fromarray(image.astype(np.uint8), mode='L')
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            # RGB image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image.astype(np.uint8))
        elif image.shape[2] == 4:
            # RGBA image
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba_image.astype(np.uint8))

    return Image.fromarray(image.astype(np.uint8))

# =============================================================================
# === PIXEL ===
# =============================================================================

def pixel_eval(color: TYPE_PIXEL,
            mode: EnumImageType=EnumImageType.RGB,
            target:EnumIntFloat=EnumIntFloat.INT,
            crunch:EnumGrayscaleCrunch=EnumGrayscaleCrunch.MEAN) -> TYPE_PIXEL:

    """Create a color by R(GB) and a target pixel type."""

    def parse_single_color(c: TYPE_PIXEL) -> TYPE_PIXEL:
        if isinstance(c, float) or c != int(c):
            c = max(0, min(1, c))
            if target == EnumIntFloat.INT:
                c = int(c * 255)

        elif isinstance(c, int):
            c = max(0, min(255, c))
            if target == EnumIntFloat.FLOAT:
                c /= 255.
        return c

    # logger.debug("{} {} {} {}", color, mode, target, crunch)

    # make sure we are an RGBA value already
    if isinstance(color, (float, int)):
        color = [parse_single_color(color)]
    elif isinstance(color, (set, tuple, list)):
        color = [parse_single_color(c) for c in color]

    if mode == EnumImageType.GRAYSCALE:
        match crunch:
            case EnumGrayscaleCrunch.LOW:
                return min(color)
            case EnumGrayscaleCrunch.HIGH:
                return max(color)
            case EnumGrayscaleCrunch.MEAN:
                return int(np.mean(color))

    elif mode == EnumImageType.RGB:
        if len(color) == 1:
            return color * 3
        if len(color) < 3:
            color += (255,) * (3 - len(color))
        return tuple(color)

    elif mode == EnumImageType.RGBA:
        if len(color) == 1:
            return color * 3 + [255]

        if len(color) < 4:
            color += (255,) * (4 - len(color))
        return (color[2], color[1], color[0], color[3])

    return color[::-1]

def pixel_convert(in_a: TYPE_IMAGE, in_b: TYPE_IMAGE) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    if in_a is not None or in_b is not None:
        if in_a is None:
            cc, w, h = channel_count(in_b)[:3]
            in_a = np.zeros((h, w, cc), dtype=np.uint8)
        if in_b is None:
            cc, w, h = channel_count(in_a)[:3]
            in_b = np.zeros((h, w, cc), dtype=np.uint8)
    return in_a, in_b

def pixel_bgr2hsv(bgr_color: TYPE_PIXEL) -> TYPE_PIXEL:
    return cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0, 0]

def pixel_hsv2bgr(hsl_color: TYPE_PIXEL) -> TYPE_PIXEL:
    return cv2.cvtColor(np.uint8([[hsl_color]]), cv2.COLOR_HSV2BGR)[0, 0]

def pixel_hsv_adjust(color:TYPE_PIXEL, hue:int=0, saturation:int=0, value:int=0,
            mod_color:bool=True, mod_sat:bool=False, mod_value:bool=False) -> TYPE_PIXEL:
    """Adjust an HSV type pixel.
    OpenCV uses... H: 0-179, S: 0-255, V: 0-255"""
    hsv = [0, 0, 0]
    hsv[0] = (color[0] + hue) % 180 if mod_color else np.clip(color[0] + hue, 0, 180)
    hsv[1] = (color[1] + saturation) % 255 if mod_sat else np.clip(color[1] + saturation, 0, 255)
    hsv[2] = (color[2] + value) % 255 if mod_value else np.clip(color[2] + value, 0, 255)
    return hsv

# =============================================================================
# === CHANNEL ===
# =============================================================================

def channel_count(image:TYPE_IMAGE) -> tuple[int, int, int, EnumImageType]:
    h, w = image.shape[:2]
    size = image.shape[2] if len(image.shape) > 2 else 1
    mode = EnumImageType.RGBA if size == 4 else EnumImageType.RGB if size == 3 else EnumImageType.GRAYSCALE
    return size, w, h, mode

def channel_add(image:TYPE_IMAGE, value: TYPE_PIXEL=255) -> TYPE_IMAGE:
    new = channel_solid(color=value, image=image)
    return np.concatenate([image, new], axis=-1)

def channel_solid(width:int=512, height:int=512, color:TYPE_PIXEL=255,
                image:Optional[TYPE_IMAGE]=None, chan:EnumImageType=EnumImageType.GRAYSCALE) -> TYPE_IMAGE:

    if image is not None:
        height, width = image.shape[:2]

    color = np.asarray(pixel_eval(color, chan))
    match chan:
        case EnumImageType.GRAYSCALE:
            return np.full((height, width, 1), color, dtype=np.uint8)

        case EnumImageType.RGB:
            return np.full((height, width, 3), color, dtype=np.uint8)

        case EnumImageType.RGBA:
            return np.full((height, width, 4), color, dtype=np.uint8)

def channel_fill(image:TYPE_IMAGE, width:int, height:int, color:TYPE_PIXEL=255) -> TYPE_IMAGE:
    """
    Fills a block of pixels with a matte or stretched to width x height.
    """

    cc, x, y, chan = channel_count(image)
    y1 = max(0, (height - y) // 2)
    y2 = min(height, y1 + y)
    x1 = max(0, (width - x) // 2)
    x2 = min(width, x1 + x)

    # crop/clip
    if y > height:
        y1 = 0
        y2 = height
    if x > width:
        x1 = 0
        x2 = width

    canvas = channel_solid(width, height, color, chan=chan)
    if cc > 1:
        canvas[y1: y2, x1: x2, :cc] = image[:y2-y1, :x2-x1, :cc]
    else:
        canvas[y1: y2, x1: x2, 0] = image[:y2-y1, :x2-x1]
    return canvas

def channel_merge(r: TYPE_IMAGE, g: TYPE_IMAGE, b: TYPE_IMAGE, a: TYPE_IMAGE,
        width: int, height: int) -> TYPE_IMAGE:

    thr, twr = r.shape[:2] if r is not None else (height, width)
    thg, twg = g.shape[:2] if g is not None else (height, width)
    thb, twb = b.shape[:2] if b is not None else (height, width)

    full = a is not None
    tha, twa = a.shape[:2] if full else (height, width)

    w = max(width, max(twa, max(twb, max(twr, twg))))
    h = max(height, max(tha, max(thb, max(thr, thg))))

    r = np.full((h, w), 0, dtype=np.uint8) if r is None else image_grayscale(r)
    g = np.full((h, w), 0, dtype=np.uint8) if g is None else image_grayscale(g)
    b = np.full((h, w), 0, dtype=np.uint8) if b is None else image_grayscale(b)

    #g = merge_channel(g, (h, w), sample)
    #b = merge_channel(b, (h, w), sample)

    if full:
        a = np.full((h, w), 0, dtype=np.uint8) if r is None else image_grayscale(a)
        # a = merge_channel(a,  (h, w), sample)
        image = cv2.merge((b, g, r, a))
    else:
        image = cv2.merge((b, g, r))
    return image

#
#
#

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
# === IMAGE ===
# =============================================================================

def image_affine_edge(image: TYPE_IMAGE, callback:object, edge:EnumEdge=EnumEdge.WRAP) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    if edge != EnumEdge.CLIP:
        image = image_edge_wrap(image, edge=edge)

    image = callback(image)

    if edge != EnumEdge.CLIP:
        image = image_crop_center(image, width, height)
    return image

def image_blend(imageA:Optional[TYPE_IMAGE]=None,
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

        img = image_scalefit(img, targetW, targetH, color, mode, sample)
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
        mask = image_scalefit(mask, w, h, color, mode, sample)
        mask = channel_fill(mask, targetW, targetH, color)
        mask = np.squeeze(mask)

    imageB[:, :, 3] = mask[:, :]
    imageA = cv2pil(imageA)
    imageB = cv2pil(imageB)
    image = blendLayers(imageA, imageB, blendOp.value, np.clip(alpha, 0, 1))
    # make sure to force the type back to uint8

    return pil2cv(image)

def image_contrast(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    cc, image, alpha = image_rgb_clean(image)
    mean_value = np.mean(image)
    image = (image - mean_value) * value + mean_value
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image_rgb_restore(image, alpha, cc == 1)

def image_crop(image: TYPE_IMAGE,
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

def image_crop_polygonal(image: TYPE_IMAGE,
                       pnt_a: TYPE_COORD, pnt_b: TYPE_COORD,
                       pnt_c: TYPE_COORD, pnt_d: TYPE_COORD,
                       color: TYPE_PIXEL=0,
                       width:int=None, height:int=None,
                       mode:EnumScaleMode=EnumScaleMode.NONE,
                       sample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:

    h, w = image.shape[:2]
    width = width if width is not None else w
    height = height if height is not None else h

    pnt_a = (int(pnt_a[0] * w), int(pnt_a[1] * h))
    pnt_b = (int(pnt_b[0] * w), int(pnt_b[1] * h))
    pnt_c = (int(pnt_c[0] * w), int(pnt_c[1] * h))
    pnt_d = (int(pnt_d[0] * w), int(pnt_d[1] * h))

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([pnt_a, pnt_b, pnt_c, pnt_d], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    x, y, w, h = cv2.boundingRect(mask[:, :])
    # image is masked off
    img = cv2.bitwise_and(image, image, mask=mask)
    img = img[y:y+h, x:x+w]
    img = image_scalefit(img, width, height, color, mode, sample)
    # canvas = np.full_like(image, color, dtype=image.dtype)
    # canvas = cv2.bitwise_and(canvas, canvas, mask=~mask)
    # img = cv2.addWeighted(canvas, 1, roi, 1, 0)
    # img = img.astype(np.uint8)

    return img, mask

def image_crop_center(image: TYPE_IMAGE, width:int=0, height:int=0) -> TYPE_IMAGE:
    h, w = image.shape[:2]
    center = (int(w * 0.5), int(h * 0.5))
    height //= 2
    width //= 2
    # logger.debug(h, w, center, width, height, center[0]-width, center[1]-height, center[0]+width, center[1]+height)
    return image[ center[1]-height:center[1]+height, center[0]-width:center[0]+width]

def image_diff(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE) -> tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, float]:
    grayA = image_grayscale(imageA)
    grayB = image_grayscale(imageB)
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    high_a = np.zeros(imageA.shape, dtype=np.uint8)
    high_b = np.zeros(imageA.shape, dtype=np.uint8)
    color = (255, 0, 0)
    for c in cnts[0]:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(high_a, (x, y), (x + w, y + h), color[::-1], -1)
        cv2.rectangle(high_b, (x, y), (x + w, y + h), color[::-1], -1)

    imageA = cv2.addWeighted(imageA, 0.5, high_a, 0.5, 1.0)
    imageB = cv2.addWeighted(imageB, 0.5, high_b, 0.5, 1.0)
    return imageA, imageB, diff, thresh, score

def image_edge_wrap(image: TYPE_IMAGE, tileX: float=1., tileY: float=1., edge:EnumEdge=EnumEdge.WRAP) -> TYPE_IMAGE:
    """TILING."""
    height, width = image.shape[:2]
    tileX = int(width * tileX) if edge in [EnumEdge.WRAP, EnumEdge.WRAPX] else 0
    tileY = int(height * tileY) if edge in [EnumEdge.WRAP, EnumEdge.WRAPY] else 0
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

def image_equalize(image:TYPE_IMAGE) -> TYPE_IMAGE:
    cc, image, alpha = image_rgb_clean(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image_rgb_restore(image, alpha, cc == 1)

def image_exposure(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    cc, image, alpha = image_rgb_clean(image)
    image = np.clip(image * value, 0, 255).astype(np.uint8)
    return image_rgb_restore(image, alpha, cc == 1)

def image_formats() -> list[str]:
    exts = Image.registered_extensions()
    return [ex for ex, f in exts.items() if f in Image.OPEN]

def image_gamma(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
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

def image_grayscale(image: TYPE_IMAGE) -> TYPE_IMAGE:
    if (cc := channel_count(image)[0]) == 1:
        return image
    if cc > 2:
        if image.dtype in [np.float16, np.float32, np.float64]:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image[:, :, 2]
    logger.error("{} {} {}", "unknown image format", cc, image.shape)
    return image

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

def image_hsv(image: TYPE_IMAGE, hue: float, saturation: float, value: float) -> TYPE_IMAGE:
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

def image_invert(image: TYPE_IMAGE, value: float) -> TYPE_IMAGE:
    value = np.clip(value, 0, 1)
    cc, image, alpha = image_rgb_clean(image)
    image = cv2.addWeighted(image, 1 - value, 255 - image, value, 0)
    return image_rgb_restore(image, alpha, cc == 1)

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

def image_load(url: str) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    """
    if img.format == 'PSD':
        images = [pil2cv(frame.copy()) for frame in ImageSequence.Iterator(img)]
        logger.debug(f"#PSD {len(images)}")
    """
    try:
        img  = cv2.imread(url, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        try:
            img = Image.open(url)
            img = ImageOps.exif_transpose(img)
            img = pil2cv(img)
        except Exception as e:
            logger.error(str(e))
            img = np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8)
            mask = np.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=np.uint8) * 255
            return img, mask

    if img.dtype != np.uint8:
        img = np.array(img / 256.0, dtype=np.float32)
    cc, width, height = channel_count(img)[:3]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    if cc == 4:
        mask = img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif cc == 2:
        img = img[:, :, 0]

    return img, mask

def image_load_data(data: str) -> TYPE_IMAGE:
    img = ImageOps.exif_transpose(data)
    img = pil2cv(img)
    #cc = channel_count(img)[0]
    #logger.debug(cc)
    #if cc == 4:
    #    img[:, :, 3] = 1. - img[:, :, 3]
    #if cc == 3:
    #    img = channel_add(img)
    return img

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

def image_mask_default(mask:TYPE_IMAGE, image:TYPE_IMAGE=None, width:int=MIN_IMAGE_SIZE, height:int=MIN_IMAGE_SIZE, mode:EnumScaleMode=EnumScaleMode.NONE) -> TYPE_IMAGE:
    if mask is None:
        if image is None or channel_count(image)[0] != 4:
            return np.ones((height, width), dtype=np.uint8) * 255
        mask = image[:, :, 3][:, :]
    else:
        mask = tensor2mask(mask)
    return image_scalefit(mask, width, height, mode=mode)

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

def image_rgb_clean(image: TYPE_IMAGE) -> tuple[int, TYPE_IMAGE, TYPE_IMAGE]:
    """Store channel, RGB, ALPHA split since most functions work with RGB."""
    alpha = None
    if (cc := channel_count(image)[0]) == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif cc == 4:
        alpha = image[:, :, 3]
        image = image[:, :, :3]  # Use slicing for consistency
    return cc, image, alpha

def image_rgb_restore(image: TYPE_IMAGE, alpha: TYPE_IMAGE, gray: bool=False) -> TYPE_IMAGE:
    if gray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if alpha is not None:
        cc = channel_count(image)[0]
        while cc < 4:
            image = channel_add(image, 0)
            cc += 1
        image[:, :, 3] = alpha
    return image

def image_rotate(image: TYPE_IMAGE, angle: float, center:TYPE_COORD=(0.5, 0.5), edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    def func_rotate(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        c = (int(width * center[0]), int(height * center[1]))
        M = cv2.getRotationMatrix2D(c, -angle, 1.0)
        return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR)

    return image_affine_edge(image, func_rotate, edge)

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

    def scale(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        w =  int(max(1, width * scale[0]))
        h =  int(max(1, height * scale[1]))
        return cv2.resize(img, (w, h), interpolation=sample.value)

    return image_affine_edge(image, scale, edge)

def image_scalefit(image: TYPE_IMAGE, width: int, height:int,
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
            return image_crop(image, (0, 0), (0, 1), (1, 1), (1, 0), width, height, color)

        case EnumScaleMode.FIT:
            return cv2.resize(image, (width, height), interpolation=sample.value)

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

def image_split(image: TYPE_IMAGE) -> tuple[TYPE_IMAGE]:
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

def image_stack(images: list[TYPE_IMAGE],
                axis:EnumOrientation=EnumOrientation.HORIZONTAL,
                stride:Optional[int]=None,
                color:TYPE_PIXEL=0,
                mode:EnumScaleMode=EnumScaleMode.NONE,
                sample:Image.Resampling=Image.Resampling.LANCZOS) -> TYPE_IMAGE:

    count = len(images)

    # CROP ALL THE IMAGES TO THE LARGEST ONE OF THE INPUT SET
    width, height = 0, 0
    for i in images:
        h, w = i.shape[:2]
        width = max(width, w)
        height = max(height, h)

    images = [channel_fill(i, width, height, color) for i in images]

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

            # logger.debug("{} {} {}", overhang, width, height, )

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

def image_stereogram(image: TYPE_IMAGE, depth: TYPE_IMAGE, divisions:int=8, mix:float=0.33, gamma:float=0.33, shift:float=1.) -> TYPE_IMAGE:
    height, width = depth.shape[:2]
    out = np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.resize(image, (width, height))
    if channel_count(image)[0] < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channel_count(depth)[0] < 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    noise = np.random.randint(0, max(1, int(gamma * 255.)), (height, width, 3), dtype=np.uint8)
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

def image_translate(image: TYPE_IMAGE, offset:TYPE_COORD=(0.0, 0.0), edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    def translate(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        scalarX = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPX] else 1.
        scalarY = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPY] else 1.

        M = np.float32([[1, 0, offset[0] * width * scalarX], [0, 1, offset[1] * height * scalarY]])
        return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR)

    return image_affine_edge(image, translate, edge)

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

def color_lut_from_image(image: TYPE_IMAGE, num_colors:int=256) -> TYPE_IMAGE:
    """Create X sized LUT from an RGB image."""
    image = cv2.resize(image, (num_colors, 1))
    return image.reshape(-1, 3).astype(np.uint8)

def color_match(image: TYPE_IMAGE, usermap: TYPE_IMAGE) -> TYPE_IMAGE:
    """Colorize one input based on the histogram matches."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    beta = cv2.cvtColor(usermap, cv2.COLOR_BGR2LAB)
    image = exposure.match_histograms(image, beta, channel_axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return image_blend(usermap, image, blendOp=BlendType.LUMINOSITY)

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

def color_match_custom_map(image: TYPE_IMAGE,
                   usermap: Optional[TYPE_IMAGE]=None,
                   colormap: int=cv2.COLORMAP_JET) -> TYPE_IMAGE:
    """Colorize one input based on custom GNU Octave/MATLAB map"""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image[:, :, 1]
    if usermap is not None:
        usermap = color_lut_from_image(usermap)
        return cv2.applyColorMap(image, usermap)
    return cv2.applyColorMap(image, colormap)

def color_match_heat_map(image: TYPE_IMAGE,
                  threshold:float=0.55,
                  colormap:int=cv2.COLORMAP_JET,
                  sigma:int=13) -> TYPE_IMAGE:
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
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    return pixel_hsv2bgr(color_a)

def color_theory_monochromatic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    sat = 255 / 5.
    val = 255 / 5.
    color_a = pixel_hsv_adjust(color, 0, -1 * sat, -1 * val, mod_sat=True, mod_value=True)
    color_b = pixel_hsv_adjust(color, 0, -2 * sat, -2 * val, mod_sat=True, mod_value=True)
    color_c = pixel_hsv_adjust(color, 0, -3 * sat, -3 * val, mod_sat=True, mod_value=True)
    color_d = pixel_hsv_adjust(color, 0, -4 * sat, -4 * val, mod_sat=True, mod_value=True)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c), pixel_hsv2bgr(color_d)

def color_theory_split_complementary(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 75, 0, 0)
    color_b = pixel_hsv_adjust(color, 105, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b)

def color_theory_analogous(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 30, 0, 0)
    color_b = pixel_hsv_adjust(color, 15, 0, 0)
    color_c = pixel_hsv_adjust(color, 165, 0, 0)
    color_d = pixel_hsv_adjust(color, 150, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c), pixel_hsv2bgr(color_d)

def color_theory_triadic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 60, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b)

def color_theory_compound(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    color_c = pixel_hsv_adjust(color, 150, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c)

def color_theory_square(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 45, 0, 0)
    color_b = pixel_hsv_adjust(color, 90, 0, 0)
    color_c = pixel_hsv_adjust(color, 135, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c)

def color_theory_tetrad_custom(color: TYPE_PIXEL, delta:int=0) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)

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
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c), pixel_hsv2bgr(color_d)

def color_theory(image: TYPE_IMAGE, custom:int=0, scheme: EnumColorTheory=EnumColorTheory.COMPLIMENTARY) -> tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE]:

    aR = aG = aB = bR = bG = bB = cR = cG = cB = dR = dG = dB = 0
    color = color_mean(image)
    match scheme:
        case EnumColorTheory.COMPLIMENTARY:
            a = color_theory_complementary(color)
            aB, aG, aR = a
        case EnumColorTheory.MONOCHROMATIC:
            a, b, c, d = color_theory_monochromatic(color)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
            dB, dG, dR = d
        case EnumColorTheory.SPLIT_COMPLIMENTARY:
            a, b = color_theory_split_complementary(color)
            aB, aG, aR = a
            bB, bG, bR = b
        case EnumColorTheory.ANALOGOUS:
            a, b, c, d = color_theory_analogous(color)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
            dB, dG, dR = d
        case EnumColorTheory.TRIADIC:
            a, b = color_theory_triadic(color)
            aB, aG, aR = a
            bB, bG, bR = b
        case EnumColorTheory.SQUARE:
            a, b, c = color_theory_square(color)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
        case EnumColorTheory.COMPOUND:
            a, b, c = color_theory_compound(color)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
        case EnumColorTheory.CUSTOM_TETRAD:
            a, b, c, d = color_theory_tetrad_custom(color, custom)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
            dB, dG, dR = d

    h, w = image.shape[:2]

    return (
        np.full((h, w, 4), [aB, aG, aR, 255], dtype=np.uint8),
        np.full((h, w, 4), [bB, bG, bR, 255], dtype=np.uint8),
        np.full((h, w, 4), [cB, cG, cR, 255], dtype=np.uint8),
        np.full((h, w, 4), [dB, dG, dR, 255], dtype=np.uint8),
        np.full((h, w, 4), color + [255], dtype=np.uint8),
    )


# =============================================================================

def cart2polar(x, y) -> tuple[Any, Any]:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta) -> tuple:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# =============================================================================

def coord_sphere(width: int, height: int, radius: float) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    theta, phi = np.meshgrid(np.linspace(0, TAU, width), np.linspace(0, np.pi, height))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    # z = radius * np.cos(phi)
    x_image = (x + 1) * (width - 1) / 2
    y_image = (y + 1) * (height - 1) / 2
    return x_image.astype(np.float32), y_image.astype(np.float32)

def coord_polar(data, origin=None) -> tuple:
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def coord_perspective(width: int, height: int, pts: list[TYPE_COORD]) -> TYPE_IMAGE:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0], pts[:, 1]])
    return cv2.getPerspectiveTransform(object_pts, pts)

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

def remap_sphere(image: TYPE_IMAGE, radius: float) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    map_x, map_y = coord_sphere(width, height, radius)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_polar(image: TYPE_IMAGE) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    map_x, map_y = coord_polar(width, height)
    map_x = map_x * width
    map_y = map_y * height
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_polar(image: TYPE_IMAGE, origin=None) -> tuple[np.ndarray, Any, Any]:
    """Re-projects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = image.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = coord_polar(image, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0]
    yi += origin[1]
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi))
    bands = []
    for band in image.T:
        zi = sp.ndimage.map_coordinates(band, coords, order=1)
        bands.append(zi.reshape((nx, ny)))
    return np.dstack(bands)
    # return output, r_i, theta_i

def remap_perspective(image: TYPE_IMAGE, pts: list) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    pts = coord_perspective(width, height, pts)
    return cv2.warpPerspective(image, pts, (width, height))

def remap_fisheye(image: TYPE_IMAGE, distort: float) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    map_x, map_y = coord_fisheye(width, height, distort)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
