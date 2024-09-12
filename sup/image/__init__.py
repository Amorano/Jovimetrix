"""
   ,  ,-.  .   , , .   , ,--. ,---. ,-.  , .   ,
   | /   \ |  /  | |\ /| |      |   |  ) |  \ /
   | |   | | /   | | V | |-     |   |-<  |   X
|  | \   / |/    | |   | |      |   |  \ |  / \
`--'  `-'  '     ' '   ' `--'   '   '  ' ' '   `

                  Image Support

     http://www.github.com/amorano/jovimetrix

     Copyright 2023 Alexander Morano (Joviex)
"""

import io
import math
import base64
import urllib
import requests
from enum import Enum
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import torch
import numpy as np
from daltonlens import simulate
from PIL import Image, ImageOps
from blendmodes.blend import BlendType, blendLayers

from loguru import logger

# =============================================================================
# === GLOBAL ===
# =============================================================================

MIN_IMAGE_SIZE: int = 32
HALFPI: float = math.pi / 2
TAU: float = math.pi * 2

IMAGE_FORMATS: List[str] = [ex for ex, f in Image.registered_extensions().items()
                            if f in Image.OPEN]

# =============================================================================
# === TYPE ===
# =============================================================================

TYPE_fCOORD2D = Tuple[float, float]
TYPE_fCOORD3D = Tuple[float, float, float]
TYPE_iCOORD2D = Tuple[int, int]
TYPE_iCOORD3D = Tuple[int, int, int]

TYPE_iRGB  = Tuple[int, int, int]
TYPE_iRGBA = Tuple[int, int, int, int]
TYPE_fRGB  = Tuple[float, float, float]
TYPE_fRGBA = Tuple[float, float, float, float]

TYPE_PIXEL = int | float | TYPE_iRGB | TYPE_iRGBA | TYPE_fRGB | TYPE_fRGBA
TYPE_IMAGE = np.ndarray | torch.Tensor
TYPE_VECTOR = TYPE_IMAGE | TYPE_PIXEL

# =============================================================================
# === ENUMERATION ===
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
# === FILE I/O ===
# =============================================================================

def image_load(url: str) -> Tuple[TYPE_IMAGE, ...]:
    try:
        img = cv2.imread(url, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"{url} could not be loaded.")

        img = image_normalize(img)
        # logger.debug(f"load image {url}: {img.ndim} {img.shape}")
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

# =============================================================================
# === CV2 CONVERSION ===
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
# === CONVERSION ===
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

def cv2tensor(image: np.ndarray, mask: bool = False) -> torch.Tensor:
    """Convert a CV2 image to a torch tensor, with handling for grayscale/mask."""

    if mask or image.ndim < 3 or (image.ndim == 3 and image.shape[2] == 1):
        image = image_mask(image)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    # logger.debug(image.shape)
    return image

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
    if tensor.ndim == 4 and tensor.shape[0] != 1:
        raise Exception("Tensor is batch of tensors")

    if tensor.ndim < 3:
        # tensor = (255.0 - tensor).unsqueeze(-1)
        tensor = tensor.unsqueeze(-1)
    else:
        tensor = tensor.squeeze()

    tensor = tensor.cpu().numpy()
    image = np.clip(255.0 * tensor, 0, 255).astype(np.uint8)

    """
    if image.shape[2] == 4:
        mask = image_mask(image)
        # we should flatten against black?
        black = np.zeros(image.shape, dtype=np.uint8)
        image = image_blend(black, image, mask)
        image = image_mask_add(image, mask)
    """
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
# === IMAGE ===
# =============================================================================
"""
These are core functions that most of the support image libraries require.
"""

def image_blend(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, mask:Optional[TYPE_IMAGE]=None,
                blendOp:BlendType=BlendType.NORMAL, alpha:float=1) -> TYPE_IMAGE:
    """Blending that will expand to the largest image pre-operation."""

    h1, w1 = imageA.shape[:2]
    h2, w2 = imageB.shape[:2]
    w = max(w1, w2)
    h = max(h1, h2)

    # logger.debug([w, h, imageA.shape, imageB.shape])
    images = []
    for img in [imageA, imageB]:
        img = image_convert(img, 4, w, h)
        images.append(img)

    imageA = images[0]
    imageB = images[1]
    old_mask = image_mask(imageB)
    if mask is None:
        mask = old_mask
    else:
        mask = image_convert(mask, 1, w, h)
        mask = mask[..., 0][:,:]
        mask = cv2.bitwise_and(mask, old_mask)

    imageA = cv2pil(imageA)
    imageB[..., 3] = mask
    imageB = cv2pil(imageB)
    alpha = np.clip(alpha, 0, 1)
    image = blendLayers(imageA, imageB, blendOp.value, alpha)
    image = pil2cv(image)

    return image_crop_center(image, w, h)

def image_convert(image: TYPE_IMAGE, channels: int, width:int=None, height:int=None,
                  matte:Tuple[int,...]=(0,0,0,0)) -> TYPE_IMAGE:
    """Force image format to a specific number of channels.

    Args:
        image (TYPE_IMAGE): Input image.
        channels (int): Desired number of channels (1, 3, or 4).
        width (int): Desired width. `None` means leave unchanged.
        height (int): Desired height. `None` means leave unchanged.
        matte (tuple): RGBA color to use as background color for transparent areas.
    Returns:
        TYPE_IMAGE: Image with the specified number of channels.
    """

    if image.ndim == 2:
        # Expand grayscale image to have a channel dimension
        image = np.expand_dims(image, -1)

    # 1, 3 or 4
    if (channels := max(1, min(4, channels))) == 2:
        channels = 1
    cc = image.shape[2] if image.ndim == 3 else 1

    if cc != channels:
        if channels == 1:
            image = image_grayscale(image)

        elif channels == 3:
            if cc == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif cc == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        elif cc == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # if there is expansion, it should be on a black canvas?
    if width is not None or height is not None:
        h, w = image.shape[:2]
        width = width or w
        height = height or h
        image = image_matte(image, matte, width, height)
        image = image_crop_center(image, width, height)

    return image

def image_crop_polygonal(image: TYPE_IMAGE, points: List[TYPE_fCOORD2D]) -> TYPE_IMAGE:
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

def image_flatten(image: List[TYPE_IMAGE], width:int=None, height:int=None,
                  mode=EnumScaleMode.MATTE,
                  sample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> TYPE_IMAGE:

    if mode == EnumScaleMode.MATTE:
        width, height = image_minmax(image)[1:]
    else:
        h, w = image[0].shape[:2]
        width = width or w
        height = height or h

    current = np.zeros((height, width, 4), dtype=np.uint8)
    for x in image:
        if mode != EnumScaleMode.MATTE:
            x = image_scalefit(x, width, height, mode, sample)
        x = image_matte(x, (0,0,0,0), width, height)
        x = image_scalefit(x, width, height, EnumScaleMode.CROP, sample)
        x = image_convert(x, 4)
        #@TODO: ADD VARIOUS COMP OPS?
        current = cv2.add(current, x)
    return current

def image_flatten_mask(image:TYPE_IMAGE, matte:Tuple=(0,0,0,255)) -> Tuple[TYPE_IMAGE, TYPE_IMAGE|None]:
    """Flatten the image with its own alpha channel, if any."""
    mask = image_mask(image)
    return image_blend(image, image, mask), mask

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

def image_matte(image: TYPE_IMAGE, color: TYPE_iRGBA=(0,0,0,255), width: int=None, height: int=None) -> TYPE_IMAGE:
    """
    Puts an RGBA image atop a colored matte expanding or clipping the image if requested.

    Args:
        image (TYPE_IMAGE): The input RGBA image.
        color (TYPE_iRGBA): The color of the matte as a tuple (R, G, B, A).
        width (int, optional): The width of the matte. Defaults to the image width.
        height (int, optional): The height of the matte. Defaults to the image height.

    Returns:
        TYPE_IMAGE: Composited RGBA image on a matte with original alpha channel.
    """

    #if image.ndim != 4 or image.shape[2] != 4:
    #    return image

    # Determine the dimensions of the image and the matte
    image_height, image_width = image.shape[:2]
    width = width or image_width
    height = height or image_height
    print(width, height)

    # Create a solid matte with the specified color
    matte = np.full((height, width, 4), color, dtype=np.uint8)

    # Extract the alpha channel from the image
    alpha = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0

    # Calculate the center position for the image on the matte
    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    if alpha is not None:
        # Place the image onto the matte using the alpha channel for blending
        for c in range(0, 3):
            matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] = \
                (1 - alpha) * matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] + \
                alpha * image[:, :, c]

        # Set the alpha channel of the matte to the maximum of the matte's and the image's alpha
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3] = \
            np.maximum(matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3], image[:, :, 3])
    else:
        image = image[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :]
    return matte

def image_matte(image: TYPE_IMAGE, color: TYPE_iRGBA= (0, 0, 0, 255), width: int = None, height: int = None) -> TYPE_IMAGE:
    """
    Puts an RGBA image atop a colored matte expanding or clipping the image if requested.

    Args:
        image (TYPE_IMAGE): The input RGBA image.
        color (TYPE_iRGBA): The color of the matte as a tuple (R, G, B, A).
        width (int, optional): The width of the matte. Defaults to the image width.
        height (int, optional): The height of the matte. Defaults to the image height.

    Returns:
        TYPE_IMAGE: Composited RGBA image on a matte with original alpha channel.
    """

    # Determine the dimensions of the image and the matte
    image_height, image_width = image.shape[:2]
    width = width or image_width
    height = height or image_height

    # Create a solid matte with the specified color
    matte = np.full((height, width, 4), color, dtype=np.uint8)

    # Calculate the center position for the image on the matte
    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    # Extract the alpha channel from the image if it's RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0

        # Blend the RGB channels using the alpha mask
        for c in range(3):  # Iterate over RGB channels
            matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] = \
                (1 - alpha) * matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] + \
                alpha * image[:, :, c]

        # Set the alpha channel to the image's alpha channel
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3] = image[:, :, 3]
    else:
        # Handle non-RGBA images (just copy the image onto the matte)
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :3] = image[:, :, :3]

    return matte

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

def image_normalize(image: TYPE_IMAGE) -> TYPE_IMAGE:
    image = image.astype(np.float32)
    img_min = np.min(image)
    img_max = np.max(image)
    if img_min == img_max:
        return np.zeros_like(image)
    image = (image - img_min) / (img_max - img_min)
    return (image * 255).astype(np.uint8)

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
