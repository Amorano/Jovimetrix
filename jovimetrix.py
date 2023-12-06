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

import os
import math
from datetime import datetime
from typing import Any, Generator

import cv2
import torch
import numpy as np
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageOps, ImageSequence

JOV_MAX_DELAY = 60.
try: JOV_MAX_DELAY = float(os.getenv("JOV_MAX_DELAY", 60.))
except: pass

# =============================================================================
# === CORE CLASSES ===
# =============================================================================

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs) -> Any:
        # If the instance does not exist, create and store it
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

# =============================================================================
# === "LOGGER" ===
# =============================================================================
class Logger(metaclass=Singleton):
    _LEVEL = int(os.getenv("JOV_LOG_LEVEL", 0))

    @classmethod
    def err(cls, *arg) -> None:
        print("\033[48;2;135;27;81;93m[JOV]\033[0m", *arg)

    @classmethod
    def warn(*arg) -> None:
        if Logger._LEVEL > 0:
            print("\033[48;2;189;135;54;93m[JOV]\033[0m", *arg)

    @classmethod
    def info(*arg) -> None:
        if Logger._LEVEL > 1:
            print("\033[48;2;54;135;27;93m[JOV]\033[0m", *arg)

    @classmethod
    def debug(*arg) -> None:
        if Logger._LEVEL > 2:
            t = datetime.now().strftime('%H:%M:%S.%f')
            print("\033[48;2;35;87;181;93m[JOV]\033[0m", t, *arg)

    @classmethod
    def spam(*arg) -> None:
        if Logger._LEVEL > 3:
            t = datetime.now().strftime('%H:%M:%S.%f')
            print("\033[48;2;55;127;201;93m[JOV]\033[0m", t, *arg)

# =============================================================================
# === CORE NODES ===
# =============================================================================

class JOVBaseNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return IT_REQUIRED

    NAME = "Jovimetrix"
    DESCRIPTION = "A Jovimetrix Node"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ()
    OUTPUT_NODE = False
    INPUT_IS_LIST = False
    FUNCTION = "run"

class JOVImageBaseNode(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("🖼️", "😷",)
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, True, )

class JOVImageInOutBaseNode(JOVBaseNode):
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("🖼️", "😷",)
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, True, )

# wildcard trick is 100% stolen from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

WILDCARD = AnyType("*")

# =============================================================================
# == SUPPORT FUNCTIONS
# =============================================================================

def zip_longest_fill(*iterables) -> Generator[tuple[Any | None, ...], Any, None]:
    iterators = [iter(iterable) for iterable in iterables]

    while True:
        values = [next(iterator, None) for iterator in iterators]

        # Check if all iterators are exhausted
        if all(value is None for value in values):
            break

        # Fill in the last values of exhausted iterators with their own last values
        for i, _ in enumerate(iterators):
            if values[i] is None:
                iterator_copy = iter(iterables[i])
                while True:
                    current_value = next(iterator_copy, None)
                    if current_value is None:
                        break
                    values[i] = current_value

        yield tuple(values)

def deep_merge_dict(*dicts: dict) -> dict:
    """
    Deep merge multiple dictionaries recursively.
    """
    def _deep_merge(d1, d2) -> Any | dict:
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            return d2

        merged_dict = d1.copy()

        for key in d2:
            if key in merged_dict:
                if isinstance(merged_dict[key], dict) and isinstance(d2[key], dict):
                    merged_dict[key] = _deep_merge(merged_dict[key], d2[key])
                elif isinstance(merged_dict[key], list) and isinstance(d2[key], list):
                    merged_dict[key].extend(d2[key])
                else:
                    merged_dict[key] = d2[key]
            else:
                merged_dict[key] = d2[key]
        return merged_dict

    merged = {}
    for d in dicts:
        merged = _deep_merge(merged, d)
    return merged

def mergePNGMeta(root: str, target: str) -> None:
    for r, _, fs in os.walk(root):
        for f in fs:
            f, ext = os.path.splitext(f)
            if ext != '.json':
                continue

            img = f"{r}/{f}.png"
            if not os.path.isfile(img):
                continue

            fn = f"{r}/{f}.json"
            with open(fn, "r", encoding="utf-8") as out:
                data = out.read()

            out = f"{target}/{f}.png"
            with Image.open(img) as image:
                metadata = PngInfo()
                for i in image.text:
                    if i == 'workflow':
                        continue
                    metadata.add_text(i, str(image.text[i]))
                metadata.add_text("workflow", data.encode('utf-8'))
                image.save(out, pnginfo=metadata)
                Logger.info(f"wrote {f} ==> {out}")

def grid_make(data: list[object]) -> list[object]:
    size = len(data)
    grid = int(math.sqrt(size))
    if grid * grid < size:
        grid += 1
    if grid < 1:
        return [], 0, 0

    rows = size // grid
    if size % grid != 0:
        rows += 1

    ret = []
    cols = 0
    for j in range(rows):
        end = min((j + 1) * grid, len(data))
        cols = max(cols, end - j * grid)
        d = [data[i] for i in range(j * grid, end)]
        ret.append(d)
    return ret, cols, rows

# =============================================================================
# === IMAGE I/O ===
# =============================================================================

def load_psd(image) -> list:
    layers=[]
    Logger.debug("load_psd", f"{image.format}")
    if image.format=='PSD':
        layers = [frame.copy() for frame in ImageSequence.Iterator(image)]
        Logger.debug("load_psd", f"#PSD {len(layers)}")
    else:
        image = ImageOps.exif_transpose(image)

    layers.append(image)
    return layers

def load_image(fp, white_bg=False) -> list:
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
# === MATRIX SUPPORT ===
# =============================================================================

def tensor2pil(tensor: torch.Tensor) -> Image:
    """Torch Tensor to PIL Image."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

def tensor2cv(tensor: torch.Tensor) -> np.ndarray[np.uint8]:
    """Torch Tensor to CV2 Matrix."""
    tensor = np.clip(255 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    if len(tensor.shape) > 2 and tensor.shape[2] > 3:
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
    if len(tensor.shape) > 2 and tensor.shape[2] > 3:
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


def cv2tensor(image: np.ndarray) -> torch.Tensor:
    """CV2 Matrix to Torch Tensor."""
    if len(image.shape) > 2 and image.shape[2] > 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA).astype(np.float32)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(image / 255.0).unsqueeze(0)

def cv2mask(image: np.ndarray) -> torch.Tensor:
    """CV2 to Torch Tensor (Mask)."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return torch.from_numpy(image / 255.0).unsqueeze(0)

def cv2pil(image: np.ndarray) -> Image:
    """CV2 Matrix to PIL."""
    if len(image.shape) > 2 and image.shape[2] > 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# =============================================================================
# === GLOBALS ===
# =============================================================================

MIN_WIDTH = MIN_HEIGHT = 256

IT_REQUIRED = {
    "required": {}
}

IT_IMAGE = {
    "required": {
        "image": ("IMAGE", ),
    }}

IT_PIXELS = {
    "optional": {
        "pixels": (WILDCARD, {}),
    }}

IT_PIXEL2 = {
    "optional": {
        "pixelA": (WILDCARD, {}),
        "pixelB": (WILDCARD, {}),
    }}

IT_WH = {
    "optional": {
        "width": ("INT", {"default": MIN_WIDTH, "min": 1, "max": 8192, "step": 1}),
        "height": ("INT", {"default": MIN_HEIGHT, "min": 1, "max": 8192, "step": 1}),
    }}

IT_SCALEMODE = {
    "optional": {
        "mode": (["NONE", "FIT", "CROP", "ASPECT"], {"default": "NONE"}),
    }}

IT_TRANS = {
    "optional": {
        "offsetX": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01}),
        "offsetY": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01}),
    }}

IT_ROT = {
    "optional": {
        "angle": ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 1}),
    }}

IT_SCALE = {
    "optional": {
        "sizeX": ("FLOAT", {"default": 1, "min": 0.01, "max": 2., "step": 0.01}),
        "sizeY": ("FLOAT", {"default": 1, "min": 0.01, "max": 2., "step": 0.01}),
    }}

IT_TILE = {
    "optional": {
        "tileX": ("INT", {"default": 1, "min": 1, "step": 1}),
        "tileY": ("INT", {"default": 1, "min": 1, "step": 1}),
    }}

IT_EDGE = {
    "optional": {
        "edge": (["CLIP", "WRAP", "WRAPX", "WRAPY"], {"default": "CLIP"}),
    }}

IT_INVERT = {
    "optional": {
        "invert": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
    }}

IT_COLOR = {
    "optional": {
        "R": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        "G": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        "B": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
    }}

IT_ORIENT = {
    "optional": {
        "orient": (["NORMAL", "FLIPX", "FLIPY", "FLIPXY"], {"default": "NORMAL"}),
    }}

IT_CAM = {
    "optional": {
        "zoom": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0}),
    }}

IT_TRS = deep_merge_dict(IT_TRANS, IT_ROT, IT_SCALE)
IT_WHMODE = deep_merge_dict(IT_WH, IT_SCALEMODE)

# =============================================================================
# === EXAMPLE MAKER ===
# =============================================================================

if __name__ == "__main__":
    mergePNGMeta('../../pysssss-workflows', './flow')
