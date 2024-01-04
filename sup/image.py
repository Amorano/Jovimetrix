"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Image Support
"""

import base64
import urllib
from enum import Enum
from io import BytesIO
from typing import Optional

import cv2
import torch
import numpy as np
import requests
from PIL import Image, ImageOps, ImageSequence

from Jovimetrix import MIN_IMAGE_SIZE, grid_make, deep_merge_dict, Logger, Lexicon, \
    IT_WH, TYPE_IMAGE, TYPE_PIXEL

# =============================================================================
# === ENUM GLOBALS ===
# =============================================================================

class EnumImageType(Enum):
    GRAYSCALE = 0
    RGB = 1
    RGBA = 2

class EnumScaleMode(Enum):
    NONE = 0
    FIT = 1
    CROP = 2
    ASPECT = 3

class EnumOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    GRID = 2

class EnumScaleMode(Enum):
    NONE = 0
    FIT = 1
    CROP = 2
    ASPECT = 3

class EnumEdge(Enum):
    CLIP = 1
    WRAP = 2
    WRAPX = 3
    WRAPY = 4

class EnumMirrorMode(Enum):
    NONE = -1
    X = 0
    Y = 1
    XY = 2
    YX = 3

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

class EnumGrayscaleCrunch(Enum):
    LOW = 0
    HIGH = 1
    MEAN = 2

class EnumIntFloat(Enum):
    FLOAT = 0
    INT = 1

# =============================================================================
# === NODE SUPPORT ===
# =============================================================================

IT_SAMPLE = {"optional": {
    Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
}}

IT_SCALEMODE = {"optional": {
    Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
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

def b64_2_tensor(base64str: str) -> tuple[torch.Tensor, torch.Tensor]:
    img = base64.b64decode(base64str)
    img = Image.open(BytesIO(img))
    return image_load_data(img)

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
    if image.mode == 'RGBA':
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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
        return Image.fromarray(image, mode='L')
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            # RGB image
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif image.shape[2] == 4:
            # RGBA image
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))

    # Default: return as-is
    return Image.fromarray(image)

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

    Logger.spam(color, mode, target, crunch)

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
        return color[::-1]

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
            cc, _, w, h = channel_count(in_b)
            in_a = np.zeros((h, w, cc), dtype=np.uint8)
        if in_b is None:
            cc, _, w, h = channel_count(in_a)
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

def channel_count(image:TYPE_IMAGE) -> tuple[int, EnumImageType, int, int]:
    h, w = image.shape[:2]
    size = image.shape[2] if len(image.shape) > 2 else 1
    mode = EnumImageType.RGBA if size == 4 else EnumImageType.RGB if size == 3 else EnumImageType.GRAYSCALE
    return size, mode, w, h

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

    cc, chan, x, y = channel_count(image)
    canvas = channel_solid(width, height, color, chan=chan)
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

    canvas[y1: y2, x1: x2, :cc] = image[:y2-y1, :x2-x1, :cc]
    return canvas

# =============================================================================
# === IMAGE ===
# =============================================================================

def image_load_data(data: str) -> TYPE_IMAGE:
    img = ImageOps.exif_transpose(data)
    img = pil2cv(img)
    cc = channel_count(img)[0]
    if cc == 4:
        img[:, :, 3] = 1. - img[:, :, 3]
    elif cc == 3:
        img = channel_add(img, 0)
    return img

def image_load(url: str) -> list[TYPE_IMAGE]:
    images = []
    try:
        img = Image.open(url)
    except Exception as e:
        Logger.err(str(e))
        return [np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=np.uint8)]

    if img.format == 'PSD':
        images = [pil2cv(frame.copy()) for frame in ImageSequence.Iterator(img)]
        Logger.debug("load_psd", f"#PSD {len(images)}")
    else:
        images = [image_load_data(img)]
    return images

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
            Logger.err(str(e))

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

def image_stack(images: list[TYPE_IMAGE],
                axis:EnumOrientation=EnumOrientation.HORIZONTAL,
                stride:Optional[int]=None,
                color:TYPE_PIXEL=0.,
                mode:EnumScaleMode=EnumScaleMode.NONE,
                sample:Image.Resampling=Image.Resampling.LANCZOS) -> TYPE_IMAGE:

    color = (
        (color[0] if color is not None else 1) * 255,
        (color[1] if color is not None else 1) * 255,
        (color[2] if color is not None else 1) * 255
    )

    count = len(images)

    # CROP ALL THE IMAGES TO THE LARGEST ONE OF THE INPUT SET
    width, height = 0, 0
    for i in images:
        w, h = i.shape[:2]
        width = max(width, w)
        height = max(height, h)

    # center = (width // 2, height // 2)
    images = [channel_fill(i, width, height, color, mode, sample) for i in images]

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

            # Logger.debug('image_stack', overhang, width, height, )

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

def image_grayscale(image: TYPE_IMAGE) -> TYPE_IMAGE:
    if (cc := channel_count(image)[0]) == 1:
        return image
    elif cc > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image[:, :, 2]
    Logger.err("unknown image format", cc, image.shape)
    return image

def image_split(image: TYPE_IMAGE) -> tuple[TYPE_IMAGE]:
    cc, _, w, h = channel_count(image)
    if cc == 4:
        b, g, r, a = cv2.split(image)
    elif cc == 3:
        b, g, r = cv2.split(image)
        a = np.full((h, w), 255, dtype=np.uint8)
    else:
        r = g = b = image
        a = np.full((h, w), 255, dtype=np.uint8)
    return r, g, b, a

def image_merge(r: TYPE_IMAGE, g: TYPE_IMAGE, b: TYPE_IMAGE, a: TYPE_IMAGE,
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

# =============================================================================
# === TEST ===
# =============================================================================

def testColorConvert() -> None:
    Logger.debug(1, pixel_eval(1., EnumImageType.RGBA))
    Logger.debug("1, 1", pixel_eval((1., 1.), EnumImageType.RGBA))
    Logger.debug("1., 1., 1., 1.", pixel_eval((1., 1., 1., 1.), EnumImageType.GRAYSCALE))
    Logger.debug(pixel_eval((255, 128, 100), EnumImageType.GRAYSCALE))
    Logger.debug(pixel_eval((255, 128, 0), EnumImageType.GRAYSCALE))
    Logger.debug(pixel_eval(255))

if __name__ == "__main__":
    testColorConvert()