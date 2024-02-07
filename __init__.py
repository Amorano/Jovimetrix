"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

              Animation, Image Compositing & Procedural Creation
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

@title: Jovimetrix
@category: Compositing
@tags: adjust, animate, audio, compose, compositing, composition, device, flow,
video, mask, shape, webcam, audio, animation, logic
@description: Webcams, GLSL shader, Media Streaming, Tick animation, Image manipulation,
Polygonal shapes, MIDI, MP3/WAVE, Flow Logic
@author: amorano
@reference: https://github.com/Amorano/Jovimetrix
@node list:
    AdjustNode, ColorMatchNode, FindEdgeNode, HSVNode, LevelsNode, ThresholdNode,
    TickNode, WaveGeneratorNode,
    GraphWaveNode,
    ConversionNode, CalcUnaryOPNode, CalcBinaryOPNode, ValueNode
    TransformNode, BlendNode, PixelSplitNode, PixelMergeNode, MergeNode, CropNode, ColorTheoryNode,
    ConstantNode, ShapeNode, TextNode, GLSLNode,
    StreamReaderNode, StreamWriterNode, MIDIMessageNode, MIDIReaderNode, MIDIFilterEZNode, MIDIFilterNode,
    DelayNode, HoldValueNode, ComparisonNode, IfThenElseNode
    AkashicNode, ValueGraphNode, RerouteNode, ExportNode, QueueNode
@version: 0.9999999999999
"""

import os
import sys
import time
import json
import shutil
import inspect
import importlib
from pathlib import Path
from typing import Any, Optional, Tuple, Union

try:
    from server import PromptServer
    from aiohttp import web
except:
    pass

import torch
import numpy as np
from loguru import logger

from Jovimetrix.sup.lexicon import Lexicon

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
JOV_GLSL = ROOT / 'res' / 'glsl'

JOV_LOG_LEVEL = os.getenv("JOV_LOG_LEVEL", "WARNING")
logger.configure(handlers=[{"sink": sys.stdout, "level": JOV_LOG_LEVEL}])

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
# === THERE CAN BE ONLY ONE ===
# =============================================================================

class Singleton(type):
    _instances = {}

    def __call__(cls, *arg, **kw) -> Any:
        # If the instance does not exist, create and store it
        if cls not in cls._instances:
            instance = super().__call__(*arg, **kw)
            cls._instances[cls] = instance
        return cls._instances[cls]

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
    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = False
    FUNCTION = "run"

class JOVImageSimple(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE,)
    OUTPUT_IS_LIST = (True, )

class JOVImageMultiple(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.MASK,)
    OUTPUT_IS_LIST = (True, True, )

# wildcard trick is 100% stolen from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

WILDCARD = AnyType("*")

# =============================================================================
# == API RESPONSE
# =============================================================================

class TimedOutException(Exception): pass

class ComfyAPIMessage:
    # STASH = {}
    MESSAGE = {}

    #@classmethod
    #def send(cls, id, message) -> None:
        #cls.MESSAGE[str(id)] = message

    @classmethod
    def poll(cls, _id, period=0.01, timeout=3) -> Any:
        _t = time.monotonic()
        sid = str(_id)
        while not (sid in cls.MESSAGE) and time.monotonic() - _t < timeout:
            time.sleep(period)

        # logger.debug(sid)
        # logger.debug(cls.MESSAGE)

        if not (sid in cls.MESSAGE):
            raise TimedOutException

        dat = cls.MESSAGE.pop(sid)
        # logger.debug(dat)
        return dat

try:
    @PromptServer.instance.routes.post("/jovimetrix/message")
    async def jovimetrix_message(request) -> Any:
        json_data = await request.json()
        did = json_data.get("id", None)
        ComfyAPIMessage.MESSAGE[str(did)] = json_data
        # logger.debug(ComfyAPIMessage.MESSAGE[did])
        return web.json_response()

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
            logger.error("bad config {}", json_data)
            return

        global JOV_CONFIG
        update_nested_dict(JOV_CONFIG, did, value)
        # logger.debug("{} {}", did, value)
        with open(JOV_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(JOV_CONFIG, f, indent=4)
        return web.json_response(json_data)

    @PromptServer.instance.routes.post("/jovimetrix/config/clear")
    async def jovimetrix_config_post(request) -> Any:
        json_data = await request.json()
        name = json_data['name']
        # logger.debug(name)
        global JOV_CONFIG
        try:
            del JOV_CONFIG['color'][name]
        except KeyError as _:
            pass
        with open(JOV_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(JOV_CONFIG, f)
        return web.json_response(json_data)

except Exception as e:
    logger.error(e)

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

# =============================================================================
# === GLOBALS ===
# =============================================================================

MIN_IMAGE_SIZE = 32

IT_REQUIRED = { "required": {} }

IT_PIXEL = { "optional": {
    Lexicon.PIXEL: (WILDCARD, {}),
}}

IT_DEPTH = { "optional": {
    Lexicon.DEPTH: (WILDCARD, {}),
}}

IT_MASK = { "optional": {
    Lexicon.MASK: (WILDCARD, {}),
}}

IT_PIXEL_MASK = {"optional": {
    Lexicon.PIXEL: (WILDCARD, {}),
    Lexicon.MASK: (WILDCARD, {}),
}}

IT_PIXEL2 = {"optional": {
    Lexicon.PIXEL_A: (WILDCARD, {}),
    Lexicon.PIXEL_B: (WILDCARD, {}),
}}

IT_PIXEL2_MASK = {"optional": {
    Lexicon.PIXEL_A: (WILDCARD, {}),
    Lexicon.PIXEL_B: (WILDCARD, {}),
    Lexicon.MASK: (WILDCARD, {}),
}}

IT_PASS_IN = {"optional": {
    Lexicon.PASS_IN: (WILDCARD, {}),
}}

IT_WH = {"optional": {
    Lexicon.WH: ("VEC2", {"default": (512, 512), "min": MIN_IMAGE_SIZE, "max": 8192, "step": 1, "label": [Lexicon.W, Lexicon.H]})
}}

IT_TRANS = {"optional": {
    Lexicon.XY: ("VEC2", {"default": (0., 0.,), "min": -1, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.X, Lexicon.Y]})
}}

IT_ROT = {"optional": {
    Lexicon.ANGLE: ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 0.01, "precision": 4, "round": 0.00001}),
}}

IT_SCALE = {"optional": {
    Lexicon.SIZE: ("VEC2", {"default": (1., 1.), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.X, Lexicon.Y]})
}}

IT_FLIP = {"optional": {
    Lexicon.FLIP: ("BOOLEAN", {"default": False}),
}}

IT_INVERT = {"optional": {
    Lexicon.INVERT: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01, "round": 0.00001, "precision": 4})
}}

IT_AB = {"optional": {
    Lexicon.IN_A: (WILDCARD, {"default": None}),
    Lexicon.IN_B: (WILDCARD, {"default": None})
}}

IT_XY = { "optional": {
    Lexicon.XY: ("VEC2", {"default": (0, 0), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.X, Lexicon.Y]})
}}

IT_XYZ = {"optional": {
    Lexicon.XYZ: ("VEC3", {"default": (0, 0, 0), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.X, Lexicon.Y, Lexicon.Z]})
}}

IT_XYZW = {"optional": {
    Lexicon.XYZW: ("VEC4", {"default": (0, 0, 0, 1), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W]})
}}

IT_RGB = {"optional": {
    Lexicon.RGB: ("VEC3", {"default": (0, 0, 0), "min": 0, "max": 255, "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B]})
}}

IT_RGBA = {"optional": {
    Lexicon.RGB_A: ("VEC4", {"default": (0, 0, 0, 255), "min": 0, "max": 255, "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A]})
}}

IT_RGBA_B = { "optional": {
    Lexicon.RGB_B: ("VEC4", {"default": (0, 0, 0, 255), "min": 0, "max": 255, "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A]})
}}

IT_RGBA_IMAGE = { "optional": {
    Lexicon.R: (WILDCARD, {}),
    Lexicon.G: (WILDCARD, {}),
    Lexicon.B: (WILDCARD, {}),
    Lexicon.A: (WILDCARD, {}),
}}

IT_HSV = { "optional": {
    Lexicon.HSV: ("VEC3",{"default": (0, 1, 1), "min": 0, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.H, Lexicon.S, Lexicon.V]})
}}

IT_GAMMA = {"optional": {
    Lexicon.GAMMA: ("FLOAT", {"default": 1, "min": 0.00001, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001})
}}

IT_CONTRAST = {"optional": {
    Lexicon.CONTRAST: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001})
}}

IT_BBOX = {"optional": {
    Lexicon.BBOX: ("VEC4", {"default": (0, 0, 1, 1), "min": 0, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]})
}}

IT_BBOX_FULL = {"optional": {
    Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "min": 0, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
    Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "min": 0, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]})
}}

IT_LOHI = {"optional": {
    Lexicon.LOHI: ("VEC2", {"default": (0, 1), "min": 0, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.LO, Lexicon.HI]})
}}

IT_LMH = {"optional": {
    Lexicon.LMH: ("VEC3", {"default": (0, 0.5, 1), "min": 0, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.LO, Lexicon.MID, Lexicon.HI]})
}}

IT_TIME = {"optional": {
    Lexicon.TIME: ("FLOAT", {"default": 0, "min": 0, "step": 0.0001, "precision": 6, "round": 0.0000001})
}}

# =============================================================================
# === SESSION ===
# =============================================================================

def configLoad() -> None:
    global JOV_CONFIG
    try:
        with open(JOV_CONFIG_FILE, 'r', encoding='utf-8') as fn:
            JOV_CONFIG = json.load(fn)
    except (IOError, FileNotFoundError) as e:
        pass
    except Exception as e:
        logger.error(e)

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
                logger.warning("---> DEFAULT CONFIGURATION <---")
            except:
                raise Exception("MAJOR 😿😰😬🥟 BLUNDERCATS 🥟😬😰😿")

        for f in (ROOT / 'nodes').iterdir():
            if f.suffix != ".py" or f.stem.startswith('_'):
                continue

            try:
                module = importlib.import_module(f"Jovimetrix.nodes.{f.stem}")
            except Exception as e:
                logger.warning(f"module failed {f}")
                logger.warning(str(e))
                continue

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

            logger.info("✅ {}", module.__name__)

        # 🔗 ⚓ 📀 🍿 🎪 🐘 🤯 😱 💀 ⛓️ 🔒 🔑 🪀 🪁 🔮 🧿 🧙🏽 🧙🏽‍♀️ 🧯 🦚

        NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in Session.CLASS_MAPPINGS.items()}
        Session.CLASS_MAPPINGS.update({k: v for k, v in Session.CLASS_MAPPINGS_WIP.items()})

        NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k, _ in Session.CLASS_MAPPINGS_WIP.items()})

        Session.CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(Session.CLASS_MAPPINGS.items(),
                                                              key=lambda item: getattr(item[1], 'SORT', 0))}
        # now sort the categories...
        for c in ["CREATE", "ADJUST", "COMPOSE", "IMAGE",
                  "CALC", "ANIMATE", "FLOW", "DEVICE", "AUDIO",
                  "UTILITY", "WIP ☣️💣"]:

            prime = Session.CLASS_MAPPINGS.copy()
            for k, v in prime.items():
                if v.CATEGORY.endswith(c):
                    NODE_CLASS_MAPPINGS[k] = v
                    Session.CLASS_MAPPINGS.pop(k)
                    logger.debug("✅ {}", k)

        # anything we dont know about sort last...
        for k, v in Session.CLASS_MAPPINGS.items():
            NODE_CLASS_MAPPINGS[k] = v
            logger.debug('⁉️ {} {}', k, v)

session = Session()
