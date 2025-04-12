"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

              Animation, Image Compositing & Procedural Creation

@title: Jovimetrix
@author: Alexander G. Morano
@category: Compositing
@reference: https://github.com/Amorano/Jovimetrix
@tags: adjust, animate, compose, compositing, composition, device, flow, video,
mask, shape, animation, logic
@description: GIPHY. Animation via tick.
Wave-based parameter modulation, Math operations with
Unary and Binary support, universal Value conversion for all major
types (int, string, list, dict, Image, Mask), shape masking, image channel ops,
batch processing, dynamic bus routing. Queue & Load from URLs.
@node list:
    ConstantNode, ShapeNode, StereogramNode, StereoscopicNode, TextNode,
    AdjustNode, BlendNode, ColorBlindNode, ColorMatchNode, ColorTheoryNode, CropNode,
    FilterMaskNode, FlattenNode, GradientMapNode, PixelMergeNode, PixelSplitNode,
    PixelSwapNode, StackNode, ThresholdNode,TransformNode,
    ComparisonNode, DelayNode, LerpNode, CalcUnaryOPNode, CalcBinaryOPNode,
    StringerNode, SwizzleNode, TickNode, ValueNode, WaveGeneratorNode,
    AkashicNode, ArrayNode, ExportNode, ValueGraphNode, ImageInfoNode, QueueNode,
    QueueTooNode, RouteNode, SaveOutputNode
"""

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__author__ = """Alexander G. Morano"""
__email__ = "amorano@gmail.com"

import os
import re
import sys
import time
import json
import inspect
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Literal, Tuple, TypeAlias

import torch

from aiohttp import web
from server import PromptServer

from loguru import logger

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

ROOT = Path(__file__).resolve().parent
ROOT_COMFY = ROOT.parent.parent
ROOT_DOC = ROOT / 'res/doc'

JOV_CONFIG = {}
JOV_WEB = ROOT / 'web'

# nodes to skip on import; for online systems; skip Export, Streamreader, etc...
JOV_IGNORE_NODE = ROOT / 'ignore.txt'

logger.add(sys.stdout, level=os.getenv("JOV_LOG_LEVEL", "INFO"),
           filter=lambda record: "jovi" in record["extra"])

JOV_INTERNAL = os.getenv("JOV_INTERNAL", 'false').strip().lower() in ('true', '1', 't')

# direct the documentation output -- used to build jovimetrix-examples
JOV_INTERNAL_DOC = os.getenv("JOV_INTERNAL_DOC", str(ROOT / "_doc"))

JOV_DOCKERENV = False
try:
    with open('/proc/1/cgroup', 'rt') as f:
        content = f.read()
        JOV_DOCKERENV = any(x in content for x in ['docker', 'kubepods', 'containerd'])
except FileNotFoundError:
    pass

if JOV_DOCKERENV:
    logger.info("RUNNING IN A DOCKER")

# The object_info route data -- cached
COMFYUI_OBJ_DATA = {}

# maximum items to show in help for combo list items
JOV_LIST_MAX = 25

# HTML TEMPLATES
TEMPLATE = {}

# BAD ACTOR NODES -- GITHUB MARKDOWN HATES EMOJI -- SCREW GITHUB MARKDOWN
MARKDOWN = [
    "ADJUST", "BLEND", "CROP", "FLATTEN", "STEREOSCOPIC", "MIDI-MESSAGE",
    "MIDI-FILTER", "STREAM-WRITER"
]

# ==============================================================================
# === TYPE ===
# ==============================================================================

class AnyType(str):
    """AnyType input wildcard trick taken from pythongossss's:

    https://github.com/pythongosssss/ComfyUI-Custom-Scripts
    """
    def __ne__(self, __value: object) -> bool:
        return False

JOV_TYPE_ANY = AnyType("*")

TensorType: TypeAlias = torch.Tensor
RGBAMaskType: TypeAlias = Tuple[TensorType, ...]
InputType: TypeAlias = Dict[str, Tuple[str|List[str], Dict[str, Any]]]

# want to make explicit entries; comfy only looks for single type
JOV_TYPE_NUMBER = "BOOLEAN,FLOAT,INT"
JOV_TYPE_VECTOR = "VEC2,VEC3,VEC4,VEC2INT,VEC3INT,VEC4INT,COORD2D,COORD3D"
JOV_TYPE_NUMERICAL = f"{JOV_TYPE_NUMBER},{JOV_TYPE_VECTOR}"
JOV_TYPE_IMAGE = "IMAGE,MASK"
JOV_TYPE_FULL = f"{JOV_TYPE_NUMBER},{JOV_TYPE_IMAGE}"

JOV_TYPE_FULL = JOV_TYPE_ANY

# ==============================================================================
# === LEXICON ===
# ==============================================================================

# EMOJI OCD Support
# 🔗 ⚓ 📀 🍿 🎪 🐘 🤯 😱 💀 ⛓️ 🔒 🔑 🪀 🪁 🧿 🧯 🦚 ♻️ ⚜️ 🚮 🤲🏽 👍 ✳️ ✌🏽 ☝🏽

class LexiconMeta(type):
    def __new__(cls, name, bases, dct) -> object:
        _tooltips = {}
        for attr_name, attr_value in dct.items():
            if isinstance(attr_value, tuple):
                attr_name = attr_value[1]
                attr_value = attr_value[0]
            _tooltips[attr_value] = attr_name
        dct['_tooltipsDB'] = _tooltips
        return super().__new__(cls, name, bases, dct)

    def __getattribute__(cls, name) -> Any | None:
        parts = name.split('.')
        value = super().__getattribute__(parts[0])
        if type(value) == tuple:
            try:
                idx = int(parts[-1])
                value = value[idx]
            except:
                value = value[0]
        return value

class Lexicon(metaclass=LexiconMeta):
    A = '⬜', "Alpha"
    ABSOLUTE = 'ABSOLUTE', "Return the absolute value of the input"
    ADAPT = '🧬', "X-Men"
    ALIGN = 'ALIGN', "Top, Center or Bottom alignment"
    AMP = '🔊', "Amplitude"
    ANGLE = '📐', "Rotation Angle"
    ANY = '🔮', "Any Type"
    ANY_OUT = '🦄', "Any Type"
    API = 'API', "API URL route"
    ATTRIBUTE = 'ATTRIBUTE', "The token attribute to use for authenticating"
    AUTH = 'AUTH', "Authentication Bearer Token"
    AUTOSIZE = 'AUTOSIZE', "Scale based on Width & Height"
    AXIS = 'AXIS', "Axis"
    B = '🟦', "Blue"
    BATCH = 'BATCH', "Output as a BATCH (all images in a single Tensor) or as a LIST of images (each image processed separately)"
    BATCH_CHUNK = 'CHUNK', "How many items to put per output. Default (0) is all items"
    BATCH_MODE = 'MODE', "Make, merge, splice or split a batch or list"
    BBOX = '🔲', "Define an inner bounding box using relative coordinates [0..1] as a box region to clip."
    BI = '💙', "Blue Channel"
    BIT = '', "Numerical Bits (0 or 1)"
    BLACK = '⬛', "Black Channel"
    BLBR = 'BL-BR', "Bottom Left - Bottom Right"
    BLUR = 'BLUR', "Blur"
    BOOLEAN = '🇴', "Boolean"
    BOTTOM = '🔽', "Bottom"
    BPM = 'BPM', "The number of Beats Per Minute"
    C1 = '🔵', "Color Scheme Result 1"
    C2 = '🟡', "Color Scheme Result 2"
    C3 = '🟣', "Color Scheme Result 3"
    C4 = '⚫️', "Color Scheme Result 4"
    C5 = '⚪', "Color Scheme Result 5"
    CAMERA = '📹', "Camera"
    C = '🇨', "Image Channels"
    CHANNEL = 'CHAN', "Channel"
    COLOR = '©️', "Color Entry for Gradient"
    COLORMAP = '🇸🇨', "One of two dozen CV2 Built-in Colormap LUT (Look Up Table) Presets"
    COLORMATCH_MAP = 'MAP', "Custom image that will be transformed into a LUT or a built-in cv2 LUT"
    COLORMATCH_MODE = 'MODE', "Match colors from an image or built-in (LUT), Histogram lookups or Reinhard method"
    COLUMNS = 'COLS', "0 = Auto-Fit, >0 = Fit into N columns"
    COMP_A = '😍', "pass this data on a successful condition"
    COMP_B = '🥵', "pass this data on a failure condition"
    COMPARE = '🕵🏽‍♀️', "Comparison function. Will pass the data in 😍 on successful comparison"
    CONTRAST = '🌓', "Contrast"
    CONTROL = '🎚️', "Control"
    COUNT = 'COUNT', 'Number of things'
    CURRENT = 'CURRENT', "Current"
    DATA = '📓', "Data"
    DEFICIENCY = 'DEFICIENCY', "Type of color deficiency: Red (Protanopia), Green (Deuteranopia), Blue (Tritanopia)"
    DELAY = '✋🏽', "Delay"
    DELTA = '🔺', "Delta"
    DEPTH = 'DEPTH', "Grayscale image representing a depth map"
    DEVICE = '📟', "Device"
    DICT = '📖', "Dictionary"
    DIFF = 'DIFF', "Difference"
    DPI = 'DPI', "Use DPI mode from OS"
    EASE = 'EASE', "Easing function"
    EDGE = 'EDGE', "Clip or Wrap the Canvas Edge"
    EDGE_X = 'EDGE_X', "Clip or Wrap the Canvas Edge"
    EDGE_Y = 'EDGE_Y', "Clip or Wrap the Canvas Edge"
    ENABLE = 'ENABLE', "Enable or Disable"
    END = 'END', "End of the range"
    FALSE = '🇫', "False"
    FILEN = '💾', "File Name"
    FILTER = '🔎', "Filter"
    FIND = 'FIND', "Find"
    FIXED = 'FIXED', "Fixed"
    FLIP = '🙃', "Flip Input A and Input B with each other"
    FLOAT = '🛟', "Float"
    FOCAL = '📽️', "Focal Length"
    FOLDER = '📁', "Folder"
    FONT = 'FONT', "Available System Fonts"
    FONT_SIZE = 'SIZE', "Text Size"
    FORMAT = 'FORMAT', "Format"
    FPS = '🏎️', "Frames per second"
    FRAME = '⏹️', "Frame"
    FREQ = 'FREQ', "Frequency"
    FUNC = '⚒️', "Function"
    G = '🟩', "Green"
    GAMMA = '🔆', "Gamma"
    GI = '💚', "Green Channel"
    GLSL_CUSTOM = '🧙🏽‍♀️', "User GLSL Shader"
    GLSL_INTERNAL = '🧙🏽', "Internal GLSL Shader"
    GRADIENT = '🇲🇺', "Gradient"
    H = '🇭', "Hue"
    HI = 'HI', "High / Top of range"
    HSV = 'HSV', "Hue, Saturation and Value"
    HOLD = '⚠️', "Hold"
    IMAGE = '🖼️', "RGB-A color image with alpha channel"
    IN_A = '🅰️', "Input A"
    IN_B = '🅱️', "Input B"
    INDEX = 'INDEX', "Current item index in the Queue list"
    INT = '🔟', "Integer"
    INVERT = '🔳', "Color Inversion"
    IO = '📋', "File I/O"
    JUSTIFY = 'JUSTIFY', "How to align the text to the side margins of the canvas: Left, Right, or Centered"
    KEY = '🔑', "Key"
    LACUNARITY = 'LACUNARITY', "LACUNARITY"
    LEFT = '◀️', "Left"
    LENGTH = 'LENGTH', "Length"
    LENGTH2 = 'FULL SIZE', "All items"
    LETTER = 'LETTER', "If each letter be generated and output in a batch"
    LINEAR = '🛟', "Linear"
    LIST = '🧾', "List"
    LMH = 'LMH', "Low, Middle, High"
    LO = 'LO', "Low"
    LOHI = 'LoHi', "Low and High"
    LOOP = '🔄', "Loop"
    LUT = '😎', "Size of each output lut palette square"
    M = '🖤', "Alpha Channel"
    MARGIN = 'MARGIN', "Whitespace padding around canvas"
    MASK = '😷', "Mask or Image to use as Mask to control where adjustments are applied"
    MATTE = 'MATTE', "Background color for padding"
    MAX = 'MAX', "Maximum"
    MI = '🤍', "Alpha Channel"
    MID = 'MID', "Middle"
    MIDI = '🎛️', "Midi"
    MIRROR = '🪞', "Mirror"
    MODE = 'MODE', "Decide whether the images should be resized to fit a specific dimension. Available modes include scaling to fit within given dimensions or keeping the original size"
    MONITOR = '🖥', "Monitor"
    NORMALIZE = '0-1', "Normalize"
    NOISE = 'NOISE', "Noise"
    NOTE = '🎶', "Note"
    OCTAVES = 'OCTAVES', "OCTAVES"
    OFFSET = 'OFFSET', "Offset"
    ON = '🔛', "On"
    OPTIMIZE = 'OPT', "Optimize"
    ORIENT = '🧭', "Orientation"
    OVERWRITE = 'OVERWRITE', "Overwrite"
    PAD = 'PAD', "Padding"
    PALETTE = '🎨', "Palette"
    PARAM = 'PARAM', "Parameters"
    PASS_IN = '📥', "Pass In"
    PASS_OUT = '📤', "Pass Out"
    PATH = 'PATH', "Selection path for array element"
    PERSISTENCE = 'PERSISTENCE', "PERSISTENCE"
    PERSPECTIVE = 'POINT', "Perspective"
    PHASE = 'PHASE', "Phase"
    PIVOT = 'PIVOT', "Pivot"
    PIXEL = '👾', "Pixel Data (RGBA, RGB or Grayscale)"
    PIXEL_A = '👾A', "Pixel Data (RGBA, RGB or Grayscale)"
    PIXEL_B = '👾B', "Pixel Data (RGBA, RGB or Grayscale)"
    PREFIX = 'PREFIX', "Prefix"
    PRESET = 'PRESET', "Preset"
    PROG_VERT = 'VERTEX', "Select a vertex program to load"
    PROG_FRAG = 'FRAGMENT', "Select a fragment program to load"
    PROJECTION = 'PROJ', "Projection"
    QUALITY = 'QUALITY', "Quality"
    QUALITY_M = 'MOTION', "Motion Quality"
    QUEUE = 'Q', "Current items to process during Queue iteration."
    R = '🟥', "Red"
    RADIUS = '🅡', "Radius"
    RANDOM = 'RNG', "Random"
    RANGE = 'RANGE', "start index, ending index (0 means full length) and how many items to skip per step"
    RATE = 'RATE', "Rate"
    RECORD = '⏺', "Arm record capture from selected device"
    REGION = 'REGION', "Region"
    RECURSE = 'RECURSE', "Search within sub-directories"
    REPLACE = 'REPLACE', "String to use as replacement"
    RESET = 'RESET', "Reset"
    RGB = '🌈', "RGB (no alpha) Color"
    RGB_A = '🌈A', "RGB (no alpha) Color"
    RGBA_A = '🌈A', "RGB with Alpha Color"
    RGBA_B = '🌈B', "RGB with Alpha Color"
    RI = '❤️', "Red Channel"
    RIGHT = '▶️', "Right"
    ROTATE = '🔃', "Rotation Angle"
    ROUND = 'ROUND', "Round to the nearest decimal place, or 0 for integer mode"
    ROUTE = '🚌', "Route"
    S = '🇸', "Saturation"
    SAMPLE = '🎞️', "Method for resizing images."
    SCHEME = 'SCHEME', "Scheme"
    SEED = 'seed', "Random generator's initial value"
    SEGMENT = 'SEGMENT', "Number of parts which the input image should be split"
    SELECT = 'SELECT', "Select"
    SHAPE = 'SHAPE', "Circle, Square or Polygonal forms"
    SHIFT = 'SHIFT', "Shift"
    SIDES = 'SIDES', "Number of sides polygon has (3-100)"
    SIMULATOR = 'SIMULATOR', "Solver to use when translating to new color space"
    SIZE = '📏', "Scalar by which to scale the input"
    SKIP = 'SKIP', "Interval between segments"
    SOURCE = 'SRC', "Source"
    SPACING = 'SPACING', "Line Spacing between Text Lines"
    START = 'START', "Start of the range"
    STEP = '🦶🏽', "Steps/Stride between pulses -- useful to do odd or even batches. If set to 0 will stretch from (VAL -> LOOP) / Batch giving a linear range of values."
    STOP = 'STOP', "Halt processing"
    STRENGTH = '💪🏽', "Strength"
    STRING = '📝', "String Entry"
    STYLE = 'STYLE', "Style"
    SWAP_A = 'SWAP A', "Replace input Alpha channel with target channel or constant"
    SWAP_B = 'SWAP B', "Replace input Blue channel with target channel or constant"
    SWAP_G = 'SWAP G', "Replace input Green channel with target channel or constant"
    SWAP_R = 'SWAP R', "Replace input Red channel with target channel or constant"
    SWAP_W = 'SWAP W', "Replace input W channel with target channel or constant"
    SWAP_X = 'SWAP X', "Replace input Red channel with target channel or constant"
    SWAP_Y = 'SWAP Y', "Replace input Red channel with target channel or constant"
    SWAP_Z = 'SWAP Z', "Replace input Red channel with target channel or constant"
    THICK = 'THICK', "Thickness"
    THRESHOLD = '📉', "Threshold"
    TILE = 'TILE', "How many times to repeat the data in the X and Y"
    TIME = '🕛', "Time"
    TIMER = '⏱', "Timer"
    TLTR = 'TL-TR', "Top Left - Top Right"
    TOGGLE = 'TOGGLE', "Toggle"
    TOP = '🔼', "Top"
    TOTAL = 'TOTAL', "Total items in the current Queue List"
    TRIGGER = '⚡', "Trigger"
    TRUE = '🇹', "True"
    TYPE = '❓', "Type"
    UNKNOWN = '❔', "Unknown"
    URL = '🌐', "URL"
    V = '🇻', "Value"
    VALUE = 'VAL', "Value"
    VEC = 'VECTOR', "Compound value of type float, vec2, vec3 or vec4"
    W = '🇼', "Width"
    WAIT = '✋🏽', "Wait"
    WAVE = '♒', "Wave Function"
    WH = '🇼🇭', "Width and Height as a Vector2 (x,y)"
    WHC = '🇼🇭🇨', "Width, Height and Channel as a Vector3 (x,y,z)"
    WINDOW = '🪟', "Window"
    X = '🇽', "X"
    X_RAW = 'X', "X"
    XY = '🇽🇾', "X and Y"
    XYZ = '🇽🇾\u200c🇿', "X, Y and Z (VEC3)"
    XYZW = '🇽🇾\u200c🇿\u200c🇼', "X, Y, Z and W (VEC4)"
    Y = '🇾', "Y"
    Y_RAW = 'Y', "Y"
    Z = '🇿', "Z"
    ZOOM = '🔎', "ZOOM"

    @classmethod
    def _parse(cls, node: dict) -> Dict[str, str]:
        for cat, entry in node.items():
            if cat not in ['optional', 'required']:
                continue
            for k, v in entry.items():
                if (widget_data := v[1] if isinstance(v, (tuple, list,)) and len(v) > 1 else None) is None:
                    continue
                if (tip := widget_data.get("tooltip", None)):
                    continue
                if (tip := cls._tooltipsDB.get(k, None)) is None:
                    continue
                widget_data["tooltip"] = tip
                node[cat][k] = (v[0], widget_data)
        return node

# ==============================================================================
# === THERE CAN BE ONLY ONE ===
# ==============================================================================

class Singleton(type):
    _instances = {}

    def __call__(cls, *arg, **kw) -> Any:
        # If the instance does not exist, create and store it
        if cls not in cls._instances:
            instance = super().__call__(*arg, **kw)
            cls._instances[cls] = instance
        return cls._instances[cls]

# ==============================================================================
# === CORE NODES ===
# ==============================================================================

class JOVBaseNode:
    NOT_IDEMPOTENT = True
    RETURN_TYPES = ()
    FUNCTION = "run"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types) -> bool:
        return True

    @classmethod
    def INPUT_TYPES(cls, prompt:bool=False, extra_png:bool=False, dynprompt:bool=False) -> Dict[str, str]:
        data = {
            "optional": {},
            "required": {},
            "hidden": {
                "ident": "UNIQUE_ID"
            }
        }
        if prompt:
            data["hidden"]["prompt"] = "PROMPT"
        if extra_png:
            data["hidden"]["extra_pnginfo"] = "EXTRA_PNGINFO"

        if dynprompt:
            data["hidden"]["dynprompt"] = "DYNPROMPT"
        return data

class JOVImageNode(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    OUTPUT_TOOLTIPS = (
        "Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output.",
        "Three channel [RGB] image. There will be no alpha.",
        "Single channel mask output."
    )

def deep_merge(d1: dict, d2: dict) -> Dict[str, str]:
    """
    Deep merge multiple dictionaries recursively.

    Args:
        *dicts: Variable number of dictionaries to be merged.

    Returns:
        dict: Merged dictionary.
    """
    for key in d2:
        if key in d1:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                deep_merge(d1[key], d2[key])
            else:
                d1[key] = d2[key]
        else:
            d1[key] = d2[key]
    return d1

# ==============================================================================
# === API RESPONSE ===
# ==============================================================================

class TimedOutException(Exception): pass

class ComfyAPIMessage:
    # STASH = {}
    MESSAGE = {}

    #@classmethod
    #def send(cls, ident, message) -> None:
        #cls.MESSAGE[str(ident)] = message

    @classmethod
    def poll(cls, ident, period=0.01, timeout=3) -> Any:
        _t = time.perf_counter()
        if isinstance(ident, (set, list, tuple, )):
            ident = ident[0]
        sid = str(ident)
        logger.debug(f'sid {sid} -- {cls.MESSAGE}')
        while not (sid in cls.MESSAGE) and time.perf_counter() - _t < timeout:
            time.sleep(period)

        if not (sid in cls.MESSAGE):
            # logger.warning(f"message failed {sid}")
            raise TimedOutException
        dat = cls.MESSAGE.pop(sid)
        return dat

def comfy_api_post(route:str, ident:str, data:dict) -> None:
    data['id'] = ident
    PromptServer.instance.send_sync(route, data)

@PromptServer.instance.routes.get("/jovimetrix/message")
async def jovimetrix_message(req) -> Any:
    return web.json_response(ComfyAPIMessage.MESSAGE)

@PromptServer.instance.routes.post("/jovimetrix/message")
async def jovimetrix_message_post(req) -> Any:
    json_data = await req.json()
    logger.info(json_data)
    if (did := json_data.get("id")) is not None:
        ComfyAPIMessage.MESSAGE[str(did)] = json_data
        return web.json_response(json_data)
    return web.json_response({})

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def parse_reset(ident:str) -> int:
    try:
        data = ComfyAPIMessage.poll(ident, timeout=0.05)
        ret = data.get('cmd', None)
        return ret == 'reset'
    except TimedOutException as e:
        return -1
    except Exception as e:
        logger.error(str(e))

def configLoad(fname:Path, as_json:bool=True) -> Any | list[str] | None:
    try:
        with open(fname, 'r', encoding='utf-8') as fn:
            if as_json:
                return json.load(fn)
            return fn.read().splitlines()
    except (IOError, FileNotFoundError) as e:
        pass
    except Exception as e:
        logger.error(e)
    return []

def load_module(name: str) -> None|ModuleType:
    module = inspect.getmodule(inspect.stack()[0][0]).__name__
    module = module.replace("\\", "/")
    route = str(name).replace("\\", "/")
    try:
        module = module.split("/")[-1]
        route = route.split(f"{module}/")[1]
        route = route.split('.')[0]
        route = route.replace('/', '.')
        module = f"{module}.{route}"
        return importlib.import_module(module)
    except Exception as e:
        logger.warning(f"file failed {name}")
        logger.warning(f"module {module}")
        logger.warning(str(e))

def loader():
    global JOV_CONFIG, JOV_IGNORE_NODE, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    NODE_LIST_MAP = {}

    if JOV_IGNORE_NODE.exists():
        JOV_IGNORE_NODE = configLoad(JOV_IGNORE_NODE, False)
    else:
        JOV_IGNORE_NODE = []

    for fname in ROOT.glob('core/**/*.py'):
        if fname.stem.startswith('_'):
            continue

        if fname.stem in JOV_IGNORE_NODE or fname.stem+'.py' in JOV_IGNORE_NODE:
            logger.warning(f"💀 [IGNORED] .core.{fname.stem}")
            continue

        if (module := load_module(fname)) is None:
            continue

        # check if there is a dynamic register function....
        try:
            for class_name, class_def in module.import_dynamic():
                setattr(module, class_name, class_def)
        except Exception as e:
            pass

        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, class_object in classes:
            # assume both attrs are good enough....
            if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                if (name := class_object.NAME) in JOV_IGNORE_NODE:
                    logger.warning(f"😥 {name}")
                    continue

                name = class_object.NAME
                NODE_DISPLAY_NAME_MAPPINGS[name] = name
                NODE_CLASS_MAPPINGS[name] = class_object

                if not name.endswith(Lexicon.GLSL_CUSTOM):
                    desc = class_object.DESCRIPTION if hasattr(class_object, 'DESCRIPTION') else name
                    NODE_LIST_MAP[name] = desc.split('.')[0].strip('\n')
                else:
                    logger.debug(f"customs {name}")

    NODE_CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(NODE_CLASS_MAPPINGS.items(),
                                                            key=lambda item: getattr(item[1], 'SORT', 0))}

    keys = NODE_CLASS_MAPPINGS.keys()
    #for name in keys:
    #    logger.debug(f"✅ {name}")
    logger.info(f"{len(keys)} nodes loaded")

    # only do the list on local runs...
    if JOV_INTERNAL:
        with open(str(ROOT) + "/node_list.json", "w", encoding="utf-8") as f:
            json.dump(NODE_LIST_MAP, f, sort_keys=True, indent=4 )

# ==============================================================================
# === BOOTSTRAP ===
# ==============================================================================

loader()
