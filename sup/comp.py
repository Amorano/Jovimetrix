"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix

                    Copyright 2023 Alexander Morano (Joviex)
"""

import math
from typing import Any, Optional

import cv2
import torch
import numpy as np
from skimage import exposure
from scipy.ndimage import rotate
from blendmodes.blend import blendLayers, BlendType
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageSequence

try:
    from .util import EnumJovian, loginfo, logwarn, logerr, logdebug
except:
    from sup.util import EnumJovian, loginfo, logwarn, logerr, logdebug

TAU = 2 * np.pi

# =============================================================================
# === GLOBAL ENUMS ===
# =============================================================================
class EnumScaleMode(EnumJovian):
    NONE = 0
    FIT = 1
    CROP = 2
    ASPECT = 3

class EnumOrientation(EnumJovian):
    HORIZONTAL = 0
    VERTICAL = 1
    GRID = 2

class EnumThreshold(EnumJovian):
    BINARY = cv2.THRESH_BINARY
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO

class EnumAdaptThreshold(EnumJovian):
    ADAPT_NONE = -1
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADAPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

EnumBlendType = [str(x).split('.')[-1] for x in BlendType]
# =============================================================================
# === MATRIX SUPPORT ===
# =============================================================================

def tensor2pil(tensor: torch.Tensor) -> Image:
    """Torch Tensor to PIL Image."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

def tensor2cv(tensor: torch.Tensor) -> np.ndarray[np.uint8]:
    """Torch Tensor to CV2 Matrix."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    if tensor.ndim > 3:
        return cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(tensor, cv2.COLOR_RGBA2BGRA)

def tensor2mask(tensor: torch.Tensor) -> np.ndarray[np.uint8]:
    """Torch Tensor to CV2 Matrix."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return cv2.cvtColor(tensor, cv2.COLOR_RGB2GRAY)

def tensor2np(tensor: torch.Tensor) -> np.ndarray[np.uint8]:
    """Torch Tensor to Numpy Array."""
    return np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

def mask2cv(tensor: torch.Tensor) -> np.ndarray[np.uint8]:
    """Torch Tensor (Mask) to CV2 Matrix."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return cv2.cvtColor(tensor, cv2.COLOR_RGB2GRAY)

def mask2pil(tensor: torch.Tensor) -> Image:
    """Torch Tensor (Mask) to PIL."""
    if tensor.ndim > 2:
        tensor = tensor.squeeze(0)
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(tensor, mode='L')

def pil2tensor(image: Image) -> torch.Tensor:
    """PIL Image to Torch Tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2cv(image: Image) -> np.ndarray[np.uint8]:
    """PIL to CV2 Matrix."""
    if image.mode == 'RGBA':
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def pil2mask(image: Image) -> torch.Tensor:
    """PIL Image to Torch Tensor (Mask)."""
    image = np.array(image.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(image)

def cv2pil(image: np.ndarray) -> Image:
    """CV2 Matrix to PIL."""
    if image.ndim > 2:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def cv2tensor(image: np.ndarray) -> torch.Tensor:
    """CV2 Matrix to Torch Tensor."""
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA).astype(np.float32)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(image / 255.0).unsqueeze(0)

def cv2mask(image: np.ndarray) -> torch.Tensor:
    """CV2 to Torch Tensor (Mask)."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return torch.from_numpy(image / 255.0).unsqueeze(0)

# =============================================================================
# === IMAGE I/O ===
# =============================================================================

def load_psd(image) -> list:
    layers=[]
    logdebug(f"[load_psd] {image.format}")
    if image.format=='PSD':
        layers = [frame.copy() for frame in ImageSequence.Iterator(image)]
        logdebug(f"[load_psd] #PSD {len(layers)}")
    else:
        image = ImageOps.exif_transpose(image)

    layers.append(image)
    return layers

def load_image(fp,white_bg=False) -> list:
    im = Image.open(fp)

    #ims = load_psd(im)
    im = ImageOps.exif_transpose(im)
    ims=[im]

    images=[]
    for i in ims:
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
            if white_bg==True:
                nw = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
                image[nw == 1] = 1.0
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

        images.append({
            "image":image,
            "mask":mask
        })

    return images

# =============================================================================
# === UTILITY ===
# =============================================================================

def gray_sized(image: cv2.Mat, h:int, w:int, resample: int=0) -> cv2.Mat:
    """Force an image into Grayscale at a specific width, height."""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[0] != h or image.shape[1] != w:
        image = cv2.resize(image, (w, h), interpolation=resample)
    return image

def split(image: cv2.Mat) -> tuple[list[cv2.Mat], list[cv2.Mat]]:
    w, h, c = image.shape

    # Check if the image has an alpha channel
    if c == 4:
        b, g, r, _ = cv2.split(image)
    else:
        b, g, r = cv2.split(image)
        # a = np.zeros_like(r)

    e = np.zeros((w, h), dtype=np.uint8)

    masks = []
    images = []
    f = cv2.merge([r, r, r])
    masks.append(cv2mask(f))

    f = cv2.merge([e, e, r])
    images.append(cv2tensor(f))

    f = cv2.merge([g, g, g])
    masks.append(cv2mask(f))
    f = cv2.merge([e, g, e])
    images.append(cv2tensor(f))

    f = cv2.merge([b, b, b])
    masks.append(cv2mask(f))
    f = cv2.merge([b, e, e])
    images.append(cv2tensor(f))

    return images, masks

def merge_channel(channel, size, resample) -> cv2.Mat:
    if channel is None:
        return np.full(size, 0, dtype=np.uint8)
    return gray_sized(channel, *size[::-1], resample)

def merge(r: cv2.Mat, g: cv2.Mat, b: cv2.Mat, a: cv2.Mat, width: int,
          height: int, mode:EnumScaleMode, resample:int) -> cv2.Mat:

    thr, twr = (r.shape[0], r.shape[1]) if r is not None else (height, width)
    thg, twg = (g.shape[0], g.shape[1]) if g is not None else (height, width)
    thb, twb = (b.shape[0], b.shape[1]) if b is not None else (height, width)
    w = max(width, max(twb, max(twr, twg)))
    h = max(height, max(thb, max(thr, thg)))

    if a is None:
        a = np.full((height, width), 255, dtype=np.uint8)
    else:
        w = max(w, a.shape[1])
        h = max(h, a.shape[0])

    target_size = (w, h)

    r = merge_channel(r, target_size, resample)
    g = merge_channel(g, target_size, resample)
    b = merge_channel(b, target_size, resample)
    a = merge_channel(a, target_size, resample)

    image = cv2.merge((r, g, b))
    return geo_scalefit(image, width, height, mode, resample)

# =============================================================================
# === SHAPE FUNCTIONS ===
# =============================================================================

def sh_body(func: str, width: int, height: int, sizeX=1., sizeY=1., fill=(255, 255, 255)) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    func = getattr(d, func)
    func(xy, fill=fill)
    return image

def sh_ellipse(width: int, height: int, sizeX=1., sizeY=1., fill=None) -> Image:
    return sh_body('ellipse', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

def sh_quad(width: int, height: int, sizeX=1., sizeY=1., fill=None) -> Image:
    return sh_body('rectangle', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

def sh_polygon(width: int, height: int, size: float=1., sides: int=3, angle: float=0., fill=None) -> Image:
    fill=fill or (255, 255, 255)
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image

# =============================================================================
# === IMAGE FUNCTIONS ===
# =============================================================================

# IMAGE

def image_stack(images: list[np.ndarray[np.uint8]],
                axis:Optional[EnumOrientation]=EnumOrientation.HORIZONTAL,
                stride:Optional[int]=None,
                color:Optional[tuple[float, float, float]]=(0,0,0),
                mode:Optional[EnumScaleMode]=EnumScaleMode.NONE,
                resample:Optional[Image.Resampling]=Image.Resampling.LANCZOS) -> np.ndarray[np.uint8]:

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

    center = (width // 2, height // 2)
    images = [comp_fill(i, center, width, height, mode, resample, color) for i in images]

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

            logdebug(overhang, width, height, )

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
               raise ValueError(f"[image_stack] invalid orientation - {axis}")

    return image

# GEOMETRY

def geo_crop(image: cv2.Mat, left=None, top=None, right=None, bottom=None,
             widthT: int=None, heightT: int=None, pad:bool=False,
             color: tuple[float, float, float]=(0, 0, 0)) -> cv2.Mat:

        height, width, _ = image.shape
        left = float(np.clip(left or 0, 0, 1))
        top = float(np.clip(top or 0, 0, 1))
        right = float(np.clip(right or 1, 0, 1))
        bottom = float(np.clip(bottom or 1, 0, 1))

        if top > bottom:
             bottom, top = top, bottom

        if left > right:
             right, left = left, right

        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2 = width * (right - left)
        ch2 = height * (bottom - top)

        crop_img = image[max(0, mid_y - ch2):min(mid_y + ch2, height),
                         max(0, mid_x - cw2):min(mid_x + cw2, width)]

        widthT = (widthT if widthT is not None else width)
        heightT = (heightT if heightT is not None else height)
        if (widthT == width and heightT == height) or not pad:
            return crop_img

        img_padded = np.full((heightT, widthT, 3), color, dtype=np.uint8)

        crop_height, crop_width, _ = crop_img.shape
        h2 = heightT // 2
        w2 = widthT // 2
        ch = crop_height // 2
        cw = crop_width // 2
        y_start, y_end = max(0, h2 - ch), min(h2 + ch, heightT)
        x_start, x_end = max(0, w2 - cw), min(w2 + cw, widthT)

        y_delta = (y_end - y_start) // 2
        x_delta = (x_end - x_start) // 2
        y_start2, y_end2 = max(0, ch - y_delta), min(ch + y_delta, crop_height)
        x_start2, x_end2 = max(0, cw - x_delta), min(cw + x_delta, crop_width)

        img_padded[y_start:y_end, x_start:x_end] = crop_img[y_start2:y_end2, x_start2:x_end2]
        logdebug(f"[geo_crop] ({x_start}, {y_start})-({x_end}, {y_end}) || ({x_start2}, {y_start2})-({x_end2}, {y_end2})")
        return img_padded

def geo_edge_wrap(image: np.ndarray[np.uint8], tileX: float=1., tileY: float=1., edge: str='WRAP') -> np.ndarray[np.uint8]:
    """TILING."""
    height, width, _ = image.shape
    tileX = int(tileX * width * 0.5) if edge in ["WRAP", "WRAPX"] else 0
    tileY = int(tileY * height * 0.5) if edge in ["WRAP", "WRAPY"] else 0
    logdebug(f"[geo_edge_wrap] [{width}, {height}]  [{tileX}, {tileY}]")
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

def geo_translate(image: np.ndarray[np.uint8], offsetX: float, offsetY: float) -> np.ndarray[np.uint8]:
    """TRANSLATION."""
    height, width, _ = image.shape
    M = np.float32([[1, 0, offsetX * width], [0, 1, offsetY * height]])
    logdebug(f"[geo_translate] [{offsetX}, {offsetY}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def geo_rotate(image: np.ndarray[np.uint8], angle: float, center=(0.5 ,0.5)) -> np.ndarray[np.uint8]:
    """ROTATION."""
    height, width, _ = image.shape
    center = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    logdebug(f"[geo_rotate] [{angle}]")
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

def geo_rotate_array(image: np.ndarray[np.uint8], angle: float, clip: bool=True) -> np.ndarray[np.uint8]:
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

def geo_scalefit(image: np.ndarray[np.uint8], width: int, height:int,
                 mode:Optional[EnumScaleMode]=EnumScaleMode.NONE,
                 resample:Optional[Image.Resampling]=Image.Resampling.LANCZOS) -> np.ndarray[np.uint8]:

    logdebug(f"[geo_scalefit] {mode} [{width}x{height}] rs=({resample})")

    if mode == EnumScaleMode.ASPECT:
        h, w, _ = image.shape
        scalar = max(width, height)
        scalar /= max(w, h)
        return cv2.resize(image, None, fx=scalar, fy=scalar, interpolation=resample)

    elif mode == EnumScaleMode.CROP:
        return geo_crop(image, widthT=width, heightT=height, pad=True)

    elif mode == EnumScaleMode.FIT:
        return cv2.resize(image, (width, height), interpolation=resample)

    return image

def geo_transform(image: np.ndarray[np.uint8], offsetX: float=0., offsetY: float=0., angle: float=0.,
              sizeX: float=1., sizeY: float=1., edge:str='CLIP', widthT: int=256, heightT: int=256,
              mode:Optional[EnumScaleMode]=EnumScaleMode.NONE,
              resample:Optional[Image.Resampling]=Image.Resampling.LANCZOS) -> np.ndarray[np.uint8]:
    """Transform, Rotate and Scale followed by Tiling and then Inversion, conforming to an input wT, hT,."""
    height, width, _ = image.shape

    # SCALE
    if sizeX != 1. or sizeY != 1.:
        wx = int(width * sizeX)
        hx = int(height * sizeY)
        image = cv2.resize(image, (wx, hx), interpolation=resample)

    # ROTATION
    if angle != 0:
        image = geo_rotate(image, angle)

    # TRANSLATION
    if offsetX != 0. or offsetY != 0.:
        image = geo_translate(image, offsetX, offsetY)

    if edge != "CLIP":
        tx = ty = 0
        if edge in ["WRAP", "WRAPX"] and sizeX < 1.:
            tx = 1. / sizeX - 1
            sizeX = 1.

        if edge in ["WRAP", "WRAPY"] and sizeY < 1.:
            ty = 1. / sizeY - 1
            sizeY = 1.

        image = geo_edge_wrap(image, tx, ty)
        h, w, _ = image.shape

    # clip to original size first...
    image = geo_crop(image)
    logdebug(f"[transform] ({offsetX},{offsetY}), {angle}, ({sizeX},{sizeY}) [{width}x{height} - {mode} - {resample}]")
    return geo_scalefit(image, widthT, heightT, mode, resample)

def geo_merge(imageA: np.ndarray[np.uint8], imageB: np.ndarray[np.uint8], axis: int=0, flip: bool=False) -> np.ndarray[np.uint8]:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def geo_mirror(image: np.ndarray[np.uint8], pX: float, axis: int, invert: bool=False) -> np.ndarray[np.uint8]:
    output =  np.zeros_like(image)
    flip = cv2.flip(image, axis)
    height, width, _ = image.shape

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

    return output

# LIGHT / COLOR

def light_hsv(image: np.ndarray[np.uint8], hue: float, saturation: float, value: float) -> np.ndarray[np.uint8]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue *= 255
    image[:, :, 0] = (image[:, :, 0] + hue) % 180
    image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
    logdebug(f"[HSV] {hue} {saturation} {value}")
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def light_gamma(image: np.ndarray[np.uint8], value: float) -> np.ndarray[np.uint8]:
    logdebug(f"[GAMMA] ({value})")
    if value == 0:
        return (image * 0).astype(np.uint8)

    invGamma = 1.0 / max(0.000001, value)
    lookUpTable = np.clip(np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype(np.uint8), 0, 255)
    return cv2.LUT(image, lookUpTable)

def light_contrast(image: np.ndarray[np.uint8], value: float) -> np.ndarray[np.uint8]:
    logdebug(f"[CONTRAST] ({value})")
    mean_value = np.mean(image)
    image = (image - mean_value) * value + mean_value
    return np.clip(image, 0, 255).astype(np.uint8)

def light_exposure(image: np.ndarray[np.uint8], value: float) -> np.ndarray[np.uint8]:
    logdebug(f"[EXPOSURE] ({math.pow(2.0, value)})")
    return np.clip(image * value, 0, 255).astype(np.uint8)

def light_invert(image: np.ndarray[np.uint8], value: float) -> np.ndarray[np.uint8]:
    value = np.clip(value, 0, 255)
    inverted = np.abs(255 - image)
    return cv2.addWeighted(image, 1 - value, inverted, value, 0)

def color_match(image: np.ndarray[np.uint8],
                target: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    beta = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    image = exposure.match_histograms(image, beta, channel_axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return comp_blend(target, image, blendOp=BlendType.LUMINOSITY)

# COMP

def comp_fill(image: np.ndarray[np.uint8],
              center: tuple[int, int],
              width:int, height:int,
              mode:Optional[EnumScaleMode]=EnumScaleMode.NONE,
              resample:Optional[Image.Resampling]=Image.Resampling.LANCZOS,
              fill:tuple[float, float, float]=(0,0,0)) -> np.ndarray[np.uint8]:
    """
    Fills a block of pixels with a matte or stretched to width x height.
    """
    if mode != EnumScaleMode.NONE:
        # image = cv2.resize(image, (width, height))
        if image.ndim > 1 and image.ndim < 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            blank = np.full((height, width), fill * 255, dtype=np.uint8)
            image[:, :, 3] = blank
            # cv2.imwrite('./_res/img/_ground-mask.png', image)
        return image

    y = image.shape[0]
    x = image.shape[1]
    blank = np.full((height, width), fill * 255, dtype=np.uint8)

    if image.ndim < 2:
        blank[center[1] - y // 2: center[1] + (y + 1) // 2,
            center[0] - x // 2: center[0] + (x + 1) // 2] = image
    else:
        blank[center[1] - y // 2: center[1] + (y + 1) // 2,
            center[0] - x // 2: center[0] + (x + 1) // 2, :3] = image[:, :, :3]

    return blank

def comp_lerp(imageA: np.ndarray[np.uint8], imageB: np.ndarray[np.uint8], mask: np.ndarray=None, alpha: float=1.) -> np.ndarray[np.uint8]:

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

def comp_blend(imageA: Optional[np.ndarray[np.uint8]],
               imageB: Optional[np.ndarray[np.uint8]],
               mask: Optional[np.ndarray[np.uint8]]=None,
               blendOp: Optional[BlendType]=BlendType.NORMAL,
               alpha: Optional[float]=1,
               mode:Optional[EnumScaleMode]=EnumScaleMode.NONE,
               resample:Optional[Image.Resampling]=Image.Resampling.LANCZOS) -> np.ndarray[np.uint8]:

    # Determine the maximum sizes among imageA, imageB
    max_width = max(
        imageA.shape[1] if imageA is not None else 0,
        imageB.shape[1] if imageB is not None else 0
    )

    max_height = max(
        imageA.shape[0] if imageA is not None else 0,
        imageB.shape[0] if imageB is not None else 0
    )
    center = (max_width // 2, max_height // 2)

    height, width = imageA.shape[:2]

    if imageA.ndim < 4 or height != max_height or width != max_width:
        imageA = comp_fill(imageA, center, max_width, max_height, mode, resample)

    height, width = imageB.shape[:2]
    if imageB.ndim < 4 or height != max_height or width != max_width:
        imageB = comp_fill(imageB, center, max_width, max_height, mode, resample)

    if mask is None:
        mask = np.ones((max_height, max_width), dtype=np.uint8) * 255
    else:
        height, width = mask.shape[:2]
        if imageB.ndim < 3 or height != max_height or width != max_width:
            mask = comp_fill(mask, center, max_width, max_height, mode, resample)

    if mask.ndim > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    imageB[:, :, 3] = mask

    logdebug('[comp_blend]', imageA.shape, imageB.shape, mask.shape, blendOp, alpha, mode, resample)

    imageA = cv2pil(imageA)
    imageB = cv2pil(imageB)
    image = blendLayers(imageA, imageB, blendOp, np.clip(alpha, 0, 1))
    return pil2cv(image)

# ADJUST

def adjust_threshold(image: np.ndarray[np.uint8], threshold: float=0.5,
              mode: Optional[EnumThreshold]=EnumThreshold.BINARY,
              adapt: Optional[EnumAdaptThreshold]=EnumAdaptThreshold.ADAPT_NONE,
              block: Optional[int]=3, const: Optional[float]=0.) -> np.ndarray[np.uint8]:

    if adapt != EnumAdaptThreshold.ADAPT_NONE.value:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(gray, 255, adapt, cv2.THRESH_BINARY, block, const)
        image = cv2.multiply(gray, image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        threshold = int(threshold * 255)
        _, image = cv2.threshold(image, threshold, 255, mode)
    return image

def adjust_levels(image: torch.Tensor, black_point:int=0, white_point=255,
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

def adjust_sharpen(image: np.ndarray[np.uint8], kernel_size=None, sigma:float=1.0,
                   amount:float=1.0, threshold:float=0) -> np.ndarray[np.uint8]:
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

# MORPHOLOGY

def morph_edge_detect(image: np.ndarray[np.uint8], low: float=0.27,
                      high:float=0.6) -> np.ndarray[np.uint8]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(src=image, ksize=(3, 5), sigmaX=0.5)
    # Perform Canny edge detection
    return cv2.Canny(image, int(low * 255), int(high * 255))

def morph_emboss(image: np.ndarray[np.uint8], amount: float=1.) -> np.ndarray[np.uint8]:
    kernel = np.array([
        [-2,    -1,     0],
        [-1,    1,      1],
        [0,     1,      2]
    ]) * amount
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# KERNELS

def MEDIAN3x3(image: np.ndarray) -> np.ndarray[np.uint8]:
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

def kernel(stride: int) -> np.ndarray[np.uint8]:
    """
    Generate a kernel matrix with a specific stride.

    The kernel matrix has a size of (stride, stride) and is filled with values
    such that if i < j, the element is set to -1; if i > j, the element is set to 1.

    Parameters:
    - stride (int): The size of the square kernel matrix.

    Returns:
    - np.ndarray: The generated kernel matrix.

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
# === REMAPPING ===
# =============================================================================

def coord_sphere(width: int, height: int, radius: float) -> tuple[np.ndarray[np.uint8], np.ndarray]:
    theta, phi = np.meshgrid(np.linspace(0, TAU, width), np.linspace(0, np.pi, height))

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    # z = radius * np.cos(phi)

    x_image = (x + 1) * (width - 1) / 2
    y_image = (y + 1) * (height - 1) / 2

    return x_image.astype(np.float32), y_image.astype(np.float32)

def coord_polar(width: int, height: int) -> tuple[np.ndarray[np.uint8], np.ndarray]:
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    rho = np.sqrt((map_x - width / 2)**2 + (map_y - height / 2)**2)
    phi = np.arctan2(map_y - height / 2, map_x - width / 2)
    return rho.astype(np.float32), phi.astype(np.float32)

def coord_perspective(width: int, height: int, pts: list[tuple[int, int]]) -> np.ndarray[np.uint8]:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0] * width, pts[:, 1] * height])
    return cv2.getPerspectiveTransform(object_pts, pts)

def coord_fisheye(width: int, height: int, distortion: float) -> tuple[np.ndarray[np.uint8], np.ndarray]:
    map_x, map_y = np.meshgrid(np.linspace(0., 1., width), np.linspace(0., 1., height))

    # normalized
    xnd, ynd = (2 * map_x - 1), (2 * map_y - 1)
    rd = np.sqrt(xnd**2 + ynd**2)

    # fish-eye distortion
    condition = (dist := 1 - distortion * (rd**2)) == 0
    xdu, ydu = np.where(condition, xnd, xnd / dist), np.where(condition, ynd, ynd / dist)
    xu, yu = ((xdu + 1) * width) / 2, ((ydu + 1) * height) / 2
    return xu.astype(np.float32), yu.astype(np.float32)

def remap_sphere(image: np.ndarray[np.uint8], radius: float) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    map_x, map_y = coord_sphere(width, height, radius)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_polar(image: np.ndarray) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    map_x, map_y = coord_polar(width, height)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_perspective(image: np.ndarray[np.uint8], pts: list) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    matrix: np.ndarray = coord_perspective(width, height, pts)
    return cv2.warpPerspective(image, matrix, (width, height))

def remap_fisheye(image: np.ndarray[np.uint8], distort: float) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    map_x, map_y = coord_fisheye(width, height, distort)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# =============================================================================
# === ZE MAIN ===
# =============================================================================

def testTRS():
    image = cv2.imread("./_res/img/alpha.png")

    pts = [
        [0.1, 0.1],
        [0.7, 0.3],
        [0.9, 0.9],
        [0.1, 0.9]
    ]
    remap = [
        ('perspective', remap_perspective(image, pts)),
        ('fisheye', remap_fisheye(image, 2)),
        ('sphere', remap_sphere(image, 0.1)),
        ('sphere', remap_sphere(image, 0.5)),
        ('sphere', remap_sphere(image, 1)),
        ('sphere', remap_sphere(image, 2)),
        ('polar', remap_polar(image)),
    ]
    idx_remap = 0
    while True:
        title, image,  = remap[idx_remap]
        cv2.imshow("", image)
        print(title)
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        idx_remap = (idx_remap + 1) % len(remap)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    back = cv2.imread('./_res/img/background.png', cv2.IMREAD_UNCHANGED)
    front = cv2.imread('./_res/img/foreground.png', cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('./_res/img/mask.png', cv2.IMREAD_UNCHANGED)
    a = comp_blend(back, front, mask, blendOp=BlendType.COLOUR, alpha=1)
    cv2.imwrite('./_res/img/_ground.png', a)
