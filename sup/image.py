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
from skimage.metrics import structural_similarity as ssim
from loguru import logger

from Jovimetrix import TYPE_IMAGE, TYPE_PIXEL, IT_WH, MIN_IMAGE_SIZE
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import grid_make, deep_merge_dict

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
    FLIP_X = 10
    Y = 20
    FLIP_Y = 30
    XY = 40
    X_FLIP_Y = 50
    FLIP_XY = 60
    FLIP_X_FLIP_Y = 70

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

# =============================================================================
# === IMAGE ===
# =============================================================================

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

def image_formats() -> list[str]:
    exts = Image.registered_extensions()
    return [ex for ex, f in exts.items() if f in Image.OPEN]

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
    if cc > 2:
        if image.dtype in [np.float16, np.float32, np.float64]:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image[:, :, 2]
    logger.error("{} {} {}", "unknown image format", cc, image.shape)
    return image

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

def image_stereogram(image: TYPE_IMAGE, depth: TYPE_IMAGE, divisions:int=4, mix:float=0.5, gamma:float=1., shift:float=1.) -> TYPE_IMAGE:
    height, width = depth.shape[:2]
    out = np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.resize(image, (width, height))
    if channel_count(depth)[0] < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    noise = np.random.randint(0, max(1, int(1 * 255)), (height, width, 3), dtype=np.uint8)
    # noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    image = cv2.addWeighted(image, 1. - mix, noise, mix, gamma)

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
