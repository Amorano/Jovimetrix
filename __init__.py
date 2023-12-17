"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix

                    Copyright 2023 Alexander Morano (Joviex)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

GO NUTS; JUST TRY NOT TO DO IT IN YOUR HEAD.

@title: Jovimetrix Composition Pack
@category: Compositing
@tags: compositing, composition, video, mask, shape, webcam
@description: Procedural & Compositing. Includes a Webcam node.
@author: amorano
@reference: https://github.com/Amorano/Jovimetrix
@node list: ConstantNode, ShapeNode, PixelShaderNode, PixelShaderImageNode,
            TransformNode, TileNode, MirrorNode, ExtendNode, HSVNode, AdjustNode,
            BlendNode, ThresholdNode, ProjectionNode, StreamReadNode, StreamWriteNode,
            RouteNode, TickNode, OptionsNode
@version: 0.98
"""

from ast import Global
import os
import math
import json
import shutil
import inspect
import importlib
from pathlib import Path
from datetime import datetime
from typing import Any, List, Generator, Optional, Tuple, Union

import cv2
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence

try:
    from server import PromptServer
    from aiohttp import web
except:
    pass

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

ROOT = Path(__file__).resolve().parent
ROOT_COMFY = ROOT.parent.parent
ROOT_COMFY_WEB = ROOT_COMFY / "web" / "extensions" / "jovimetrix"

JOV_CONFIG = {}
JOV_WEB = ROOT / 'web'
JOV_DEFAULT = JOV_WEB / 'default.json'
JOV_CONFIG_FILE = JOV_WEB / 'config.json'

JOV_MAX_DELAY = 60.
try: JOV_MAX_DELAY = float(os.getenv("JOV_MAX_DELAY", 60.))
except: pass

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

class Logger(metaclass=Singleton):
    _LEVEL = int(os.getenv("JOV_LOG_LEVEL", 0))

    @classmethod
    def _raw(cls, color, who, *arg) -> None:
        if who is None:
            print(color, '[JOV]\033[0m', *arg)
        else:
            print(color, '[JOV]\033[0m', f'({who})', *arg)

    @classmethod
    def dump(cls, *arg) -> None:
        who = inspect.currentframe().f_back.f_code.co_name
        cls._raw("\033[48;2;35;127;81;93m", who, None, *arg)

    @classmethod
    def err(cls, *arg) -> None:
        who = inspect.currentframe().f_back.f_code.co_name
        cls._raw("\033[48;2;135;27;81;93m", who, None, *arg)

    @classmethod
    def warn(cls, *arg) -> None:
        if Logger._LEVEL > 0:
            who = inspect.currentframe().f_back.f_code.co_name
            cls._raw("\033[48;2;159;155;44;93m", None, *arg)

    @classmethod
    def info(cls, *arg) -> None:
        if Logger._LEVEL > 1:
            cls._raw("\033[48;2;44;115;37;93m", None, *arg)

    @classmethod
    def debug(cls, *arg) -> None:
        if Logger._LEVEL > 2:
            t = datetime.now().strftime('%H:%M:%S.%f')
            who = inspect.currentframe().f_back.f_code.co_name
            cls._raw("\033[48;2;35;87;181;93m", t, who, *arg)

    @classmethod
    def spam(cls, *arg) -> None:
        if Logger._LEVEL > 3:
            t = datetime.now().strftime('%H:%M:%S.%f')
            who = inspect.currentframe().f_back.f_code.co_name
            cls._raw("\033[48;2;35;87;181;93m", t, who, *arg)

class Session(metaclass=Singleton):
    CLASS_MAPPINGS = {}
    CLASS_MAPPINGS_WIP = {}

    @classmethod
    def ignore_files(cls, d, files) -> list[str]|None:
        return [x for x in files if x.endswith('.json') or x.endswith('.html')]

    def __init__(self, *arg, **kw) -> None:
        global JOV_CONFIG
        found = False
        if JOV_CONFIG_FILE.exists():
            configLoad()
            # is this an old config, copy default (sorry, not sorry)
            found = JOV_CONFIG.get('user', None) is not None

        if not found:
            try:
                shutil.copy2(JOV_DEFAULT, JOV_CONFIG_FILE)
                Logger.warn("---> DEFAULT CONFIGURATION <---")
            except:
                raise Exception("MAJOR 😿😰😬🥟 BLUNDERCATS 🥟😬😰😿")

        for f in (ROOT / 'nodes').iterdir():
            if f.suffix != ".py" or f.stem.startswith('_'):
                continue

            module = importlib.import_module(f"Jovimetrix.nodes.{f.stem}")
            classes = inspect.getmembers(module, inspect.isclass)
            for class_name, class_object in classes:
                # assume both attrs are good enough....
                if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                    name = class_object.NAME
                    if hasattr(class_object, 'POST'):
                        class_object.CATEGORY = "JOVIMETRIX 🔺🟩🔵/WIP ☣️💣"
                        Session.CLASS_MAPPINGS_WIP[name] = class_object
                    else:
                        Session.CLASS_MAPPINGS[name] = class_object

            Logger.info("✅", module.__name__)

        # 🔗 ⚓ 📀 🍿 🎪 🐘 🤯 😱 💀 ⛓️ 🔒 🔑 🪀 🪁 🔮 🧿 🧙🏽 🧙🏽‍♀️ 🧯 🦚

        NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in Session.CLASS_MAPPINGS.items()}
        Session.CLASS_MAPPINGS.update({k: v for k, v in Session.CLASS_MAPPINGS_WIP.items()})

        NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k, _ in Session.CLASS_MAPPINGS_WIP.items()})

        Session.CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(Session.CLASS_MAPPINGS.items(),
                                                              key=lambda item: getattr(item[1], 'SORT', 0))}
        # now sort the categories...
        for c in ["CREATE", "ADJUST", "TRANSFORM", "COMPOSE",
                  "ANIMATE", "FLOW", "DEVICE", "AUDIO",
                  "UTILITY", "WIP ☣️💣"]:

            prime = Session.CLASS_MAPPINGS.copy()
            for k, v in prime.items():
                if v.CATEGORY.endswith(c):
                    NODE_CLASS_MAPPINGS[k] = v
                    Session.CLASS_MAPPINGS.pop(k)
                    Logger.debug("✅", k)

        # anything we dont know about sort last...
        for k, v in Session.CLASS_MAPPINGS.items():
            NODE_CLASS_MAPPINGS[k] = v
            Logger.debug('⁉️', k, v)

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

#
#
#

def configLoad() -> None:
    global JOV_CONFIG
    try:
        with open(JOV_CONFIG_FILE, 'r', encoding='utf-8') as fn:
            JOV_CONFIG = json.load(fn)
    except (IOError, FileNotFoundError) as e:
        pass
    except Exception as e:
        print(e)

# =============================================================================
# == CUSTOM API RESPONSES
# =============================================================================

try:
    @PromptServer.instance.routes.get("/jovimetrix/config")
    async def jovimetrix_config(request) -> Any:
        global JOV_CONFIG
        configLoad()
        return web.json_response(JOV_CONFIG)

    @PromptServer.instance.routes.post("/jovimetrix/config")
    async def jovimetrix_config_post(request) -> Any:
        json_data = await request.json()
        did = json_data.get("id", None)
        value = json_data.get("v", None)
        if did is None or value is None:
            Logger.error("bad config", json_data)
            return

        global JOV_CONFIG
        update_nested_dict(JOV_CONFIG, did, value)
        Logger.spam(did, value)
        with open(JOV_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(JOV_CONFIG, f, indent=4)
        return web.json_response(json_data)

    @PromptServer.instance.routes.post("/jovimetrix/config/clear")
    async def jovimetrix_config_post(request) -> Any:
        json_data = await request.json()
        name = json_data['name']
        Logger.spam(name)
        global JOV_CONFIG
        try:
            del JOV_CONFIG['color'][name]
        except KeyError as _:
            pass
        with open(JOV_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(JOV_CONFIG, f)
        return web.json_response(json_data)

except Exception as e:
    print(e)
# =============================================================================
# == SUPPORT FUNCTIONS
# =============================================================================

def update_nested_dict(d, path, value) -> None:
    keys = path.split('.')
    current = d

    for key in keys[:-1]:
        current = current.setdefault(key, {})

    last_key = keys[-1]

    # Check if the key already exists
    if last_key in current and isinstance(current[last_key], dict):
        current[last_key].update(value)
    else:
        current[last_key] = value

def zip_longest_fill(*iterables: Any) -> Generator[Tuple[Any, ...], None, None]:
    """
    Zip longest with fill value.

    This function behaves like itertools.zip_longest, but it fills the values
    of exhausted iterators with their own last values instead of None.
    """
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

    Args:
        *dicts: Variable number of dictionaries to be merged.

    Returns:
        dict: Merged dictionary.
    """
    def _deep_merge(d1: Any, d2: Any) -> Any:
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

def grid_make(data: List[Any]) -> Tuple[List[List[Any]], int, int]:
    """
    Create a 2D grid from a 1D list.

    Args:
        data (List[Any]): Input data.

    Returns:
        Tuple[List[List[Any]], int, int]: A tuple containing the 2D grid, number of columns,
        and number of rows.
    """
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
# === GLOBALS ===
# =============================================================================

MIN_WIDTH = MIN_HEIGHT = 256

IT_REQUIRED = {
    "required": {}
}

IT_PIXELS = {
    "optional": {
        "👾": (WILDCARD, {}),
    }}

IT_PIXELS_REQUIRED = {
    "required": {
        "👾": (WILDCARD, {}),
    }}

IT_PIXEL2 = {
    "optional": {
        "👾A": (WILDCARD, {}),
        "👾B": (WILDCARD, {}),
    }}

IT_WH = {
    "optional": {
        "↔️": ("INT", {"default": MIN_WIDTH, "min": 1, "max": 8192, "step": 1}),
        "↕️": ("INT", {"default": MIN_HEIGHT, "min": 1, "max": 8192, "step": 1}),
    }}

IT_SCALEMODE = {
    "optional": {
        "mode": (["NONE", "FIT", "CROP", "ASPECT"], {"default": "NONE"}),
    }}

IT_TRANS = {
    "optional": {
        "🇽": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01}),
        "🇾": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01}),
    }}

IT_ROT = {
    "optional": {
        "📐": ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 1}),
    }}

IT_SCALE = {
    "optional": {
        "➡️": ("FLOAT", {"default": 1, "min": 0.01, "max": 2., "step": 0.01}),
        "⬆️": ("FLOAT", {"default": 1, "min": 0.01, "max": 2., "step": 0.01}),
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

IT_FLIP = {
    "optional": {
        "↩️": ("BOOLEAN", {"default": False}),
    }}

IT_INVERT = {
    "optional": {
        "🔳": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
    }}

IT_COLOR = {
    "optional": {
        "🟥": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        "🟩": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        "🟦": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
    }}

IT_ORIENT = {
    "optional": {
        "🔄": (["NORMAL", "FLIPX", "FLIPY", "FLIPXY"], {"default": "NORMAL"}),
    }}

IT_CAM = {
    "optional": {
        "⤴️": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0}),
    }}

IT_TRS = deep_merge_dict(IT_TRANS, IT_ROT, IT_SCALE)
IT_WHMODE = deep_merge_dict(IT_WH, IT_SCALEMODE)

session = Session()

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
