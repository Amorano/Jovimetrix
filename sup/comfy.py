"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
COMFY Support
"""

import os

import json
from pathlib import Path
from typing import Any

try:
    from server import PromptServer
    from aiohttp import web
except:
    pass

from Jovimetrix import Logger, configLoad
from Jovimetrix.sup.lexicon import Lexicon

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
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.MASK,)
    OUTPUT_IS_LIST = (True, True, )

class JOVImageInOutBaseNode(JOVImageBaseNode):
    INPUT_IS_LIST = True

# wildcard trick is 100% stolen from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

WILDCARD = AnyType("*")

# =============================================================================
# == API RESPONSE
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
    Logger.err(e)

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

IT_PIXELS = { "optional": {
    Lexicon.PIXEL: (WILDCARD, {}),
}}

IT_PIXEL2 = {"optional": {
    Lexicon.PIXEL_A: (WILDCARD, {}),
    Lexicon.PIXEL_B: (WILDCARD, {}),
}}

IT_PIXEL_MASK = {"optional": {
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
    Lexicon.OFFSET: ("VEC2", {"default": (0., 0.,), "min": -1, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y]})
}}

IT_ROT = {"optional": {
    Lexicon.ANGLE: ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 0.005, "precision": 4}),
}}

IT_SCALE = {"optional": {
    Lexicon.SIZE: ("VEC2", {"default": (1., 1.), "min": -1., "max": 1., "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y]})
}}

IT_FLIP = {"optional": {
    Lexicon.FLIP: ("BOOLEAN", {"default": False}),
}}

IT_INVERT = {"optional": {
    Lexicon.INVERT: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.005, "precision": 4})
}}

IT_AB = {"optional": {
    Lexicon.IN_A: (WILDCARD, {"default": None}),
    Lexicon.IN_B: (WILDCARD, {"default": None})
}}

IT_XY = { "optional": {
    Lexicon.XY: ("VEC2", {"default": (0, 0), "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y]})
}}

IT_XYZ = {"optional": {
    Lexicon.XYZ: ("VEC3", {"default": (0, 0, 0), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y, Lexicon.Z]})
}}

IT_XYZW = {"optional": {
    Lexicon.XYZW: ("VEC4", {"default": (0, 0, 0, 1), "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W]})
}}

IT_RGBA = {"optional": {
    Lexicon.RGBA: ("VEC4", {"default": (0, 0, 0, 255), "min": 0, "max": 255, "step": 0.0625, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A]})
}}

IT_RGBA_B = { "optional": {
    Lexicon.RGBA_B: ("VEC4", {"default": (0, 0, 0, 255), "min": 0, "max": 255, "step": 0.0625, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A]})
}}

IT_RGBA_IMAGE = { "optional": {
    Lexicon.R: (WILDCARD, {}),
    Lexicon.G: (WILDCARD, {}),
    Lexicon.B: (WILDCARD, {}),
    Lexicon.A: (WILDCARD, {}),
}}

IT_HSV = { "optional": {
    Lexicon.HSV: ("VEC3",{"default": (0, 1, 1), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.H, Lexicon.S, Lexicon.V]})
}}

IT_GAMMA = {"optional": {
    Lexicon.GAMMA: ("FLOAT", {"default": 1, "min": 0.00001, "max": 1, "step": 0.005, "precision": 4})
}}

IT_CONTRAST = {"optional": {
    Lexicon.CONTRAST: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.005, "precision": 4})
}}

IT_BBOX = {"optional": {
    Lexicon.BBOX: ("VEC4", {"default": (0, 0, 1, 1), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]})
}}

IT_BBOX_FULL = {"optional": {
    Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
    Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]})
}}

IT_LOHI = {"optional": {
    Lexicon.LOHI: ("VEC2", {"default": (0, 1), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.LO, Lexicon.HI]})
}}

IT_LMH = {"optional": {
    Lexicon.LMH: ("VEC3", {"default": (0, 0.5, 1), "min": 0, "max": 1, "step": 0.01, "precision": 4, "label": [Lexicon.LO, Lexicon.MID, Lexicon.HI]})
}}

IT_TIME = {"optional": {
    Lexicon.TIME: ("FLOAT", {"default": 0, "min": 0, "step": 0.000001, "precision": 6})
}}

IT_TRS = deep_merge_dict(IT_TRANS, IT_ROT, IT_SCALE)

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
