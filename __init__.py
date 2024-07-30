"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

              Animation, Image Compositing & Procedural Creation
                    http://www.github.com/amorano/jovimetrix

@title: Jovimetrix
@author: amorano
@category: Compositing
@reference: https://github.com/Amorano/Jovimetrix
@tags: adjust, animate, compose, compositing, composition, device, flow, video,
mask, shape, webcam, animation, logic
@description: Integrates Webcam, MIDI, Spout and GLSL shader support. Animation
via tick. Parameter manipulation with wave generator. Math operations with Unary
and Binary support. Value conversion for all major types (int, string, list,
dict, Image, Mask). Shape mask generation, image stacking and channel ops, batch
splitting, merging and randomizing, load images and video from anywhere, dynamic
bus routing with a single node, export support for GIPHY, save output anywhere!
flatten, crop, transform; check colorblindness, make stereogram or stereoscopic
images, or liner interpolate values and more.
@node list:
    ConstantNode, GLSLNode, ShapeNode, StereogramNode, StereoscopicNode, TextNode, WaveGraphNode,
    AdjustNode, ColorBlindNode, ColorMatchNode, FilterMaskNode, ThresholdNode,
    BlendNode, ColorTheoryNode, CropNode, FlattenNode, PixelMergeNode, PixelSplitNode,
    PixelSwapNode, StackNode, TransformNode
    CalcUnaryOPNode, CalcBinaryOPNode, ValueNode, ConvertNode, LerpNode, DelayNode,
    ComparisonNode,
    TickNode, WaveGeneratorNode,
    MIDIMessageNode, MIDIReaderNode, MIDIFilterEZNode, MIDIFilterNode,
    StreamReaderNode, StreamWriterNode, SpoutWriter,
    AkashicNode, ArrayNode, BatchLoadNode, DynamicNode, ValueGraphNode, ExportNode, QueueNode,
    RouteNode, SaveOutputNode
@version: 1.2.6
"""

import os
import re
import sys
import html
import time
import json
import shutil
import inspect
import textwrap
import importlib
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Tuple

try:
    from server import PromptServer
    from aiohttp import web
except:
    pass

from loguru import logger

NODE_LIST_MAP = {}
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

ROOT = Path(__file__).resolve().parent
ROOT_COMFY = ROOT.parent.parent
ROOT_DOC = ROOT / 'res/doc'

JOV_CONFIG = {}
JOV_WEB = ROOT / 'web'
JOV_DEFAULT = JOV_WEB / 'default.json'
JOV_CONFIG_FILE = JOV_WEB / 'config.json'

# nodes to skip on import; for online systems; skip Export, Streamreader, etc...
JOV_IGNORE_NODE = ROOT / 'ignore.txt'

JOV_LOG_LEVEL = os.getenv("JOV_LOG_LEVEL", "WARNING")
logger.configure(handlers=[{"sink": sys.stdout, "level": JOV_LOG_LEVEL}])

JOV_INTERNAL = os.getenv("JOV_INTERNAL", 'false').strip().lower() in ('true', '1', 't')

# direct the documentation output -- used to build jovimetrix-examples
JOV_INTERNAL_DOC = os.getenv("JOV_INTERNAL_DOC", str(ROOT / "_doc"))

# any/all documentation auto-made on request
DOCUMENTATION = {}

# maximum items to show in help for combo list items
JOV_LIST_MAX = 25

# HTML TEMPLATES
TEMPLATE = {}

# =============================================================================
# === LEXICON ===
# =============================================================================

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
    BBOX = '🔲', "Bounding box"
    BI = '💙', "Blue Channel"
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
    END = 'END', "End of the range"
    FALSE = '🇫', "False"
    FILEN = '💾', "File Name"
    FILTER = '🔎', "Filter"
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
    GRADIENT = '🇲🇺', "Gradient"
    H = '🇭', "Hue"
    HI = 'HI', "High / Top of range"
    HSV = 'HSV', "Hue, Saturation and Value"
    HOLD = '⚠️', "Hold"
    IMAGE = '🖼️', "Image"
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
    M = '🖤', "Alpha Channel"
    MARGIN = 'MARGIN', "Whitespace padding around canvas"
    MASK = '😷', "Mask or Image to use as Mask to control where adjustments are applied"
    MATTE = 'MATTE', "Define a background color for padding, if necessary. This is useful when images do not fit perfectly into the designated area and need a filler color"
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
    QUEUE = 'Q', "Queue"
    R = '🟥', "Red"
    RADIUS = '🅡', "Radius"
    RANDOM = 'RNG', "Random"
    RANGE = 'RANGE', "start index, ending index (0 means full length) and how many items to skip per step"
    RATE = 'RATE', "Rate"
    RECORD = '⏺', "Arm record capture from selected device"
    REGION = 'REGION', "Region"
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
    SAMPLE = '🎞️', "Select the method for resizing images. Options range from nearest neighbor to advanced methods like Lanczos, ensuring the best quality for the specific use case"
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
    STEP = '🦶🏽', "Step"
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
    TILE = 'TILE', "Title"
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
    def _parse(cls, node: dict, node_cls: object) -> dict:
        name_url = node_cls.NAME.split(" (JOV)")[0]
        url = name_url.replace(" ", "-")
        cat = '/'.join(node_cls.CATEGORY.split('/')[1:])
        data = {"_": f"{cat}#-{url}", "*": node_cls.NAME, "outputs": {}}
        for cat, entry in node.items():
            if cat not in ['optional', 'required', 'outputs']:
                continue
            for k, v in entry.items():
                widget_data = v[1] if isinstance(v, (tuple, list,)) and len(v) > 1 else {}
                if (tip := widget_data.get('tooltip', None)) is None:
                    if (tip := cls._tooltipsDB.get(k), None) is None:
                        logger.warning(f"no {k}")
                        continue
                if cat == "outputs":
                    data["outputs"][k] = tip
                else:
                    data[k] = tip
        if node.get("optional", None) is None:
            node["optional"] = {}
        node["optional"]["tooltips"] = ("JTOOLTIP", {"default": data})
        return node

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
    RETURN_TYPES = ()
    FUNCTION = "run"
    # instance map for caching
    INSTANCE = {}

    @classmethod
    def INPUT_TYPES(cls, prompt:bool=False, extra_png:bool=False) -> dict:
        data = {
            "required": {},
            "hidden": {
                "ident": "UNIQUE_ID"
            }
        }
        if prompt:
            data["hidden"]["prompt"] = "PROMPT"
        if extra_png:
            data["hidden"]["extra_pnginfo"] = "EXTRA_PNGINFO"
        return data

class JOVImageNode(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "outputs": {
                0: ("IMAGE", {"tooltip":"Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output."}),
                1: ("IMAGE", {"tooltip":"Three channel [RGB] image. There will be no alpha."}),
                2: ("MASK", {"tooltip":"Single channel mask output."}),
            }
        })
        return Lexicon._parse(d, cls)

# wildcard trick is 100% stolen from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

JOV_TYPE_ANY = AnyType("*")

# want to make explicit entries; comfy only looks for single type
JOV_TYPE_COMFY = "BOOLEAN,FLOAT,INT"
JOV_TYPE_VECTOR = "VEC2,VEC3,VEC4,VEC2INT,VEC3INT,VEC4INT,COORD2D"
JOV_TYPE_NUMBER = f"{JOV_TYPE_COMFY},{JOV_TYPE_VECTOR}"
JOV_TYPE_IMAGE = "IMAGE,MASK"
JOV_TYPE_FULL = f"{JOV_TYPE_NUMBER},{JOV_TYPE_IMAGE}"

JOV_TYPE_COMFY = JOV_TYPE_ANY
JOV_TYPE_VECTOR = JOV_TYPE_ANY
JOV_TYPE_NUMBER = JOV_TYPE_ANY
JOV_TYPE_IMAGE = JOV_TYPE_ANY
JOV_TYPE_FULL = JOV_TYPE_ANY

# =============================================================================
# === DOCUMENTATION SUPPORT
# =============================================================================

"""
JUDICIOUS BORROWING FROM SALT.AI DOCUMENTATION PROJECT:
https://github.com/get-salt-AI/SaltAI_Documentation_Tools
"""

def collapse_repeating_parameters(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Collapses repeating parameters like `input_blocks.0`,...`input_blocks.10` into 1 parameter `input_blocks.i`."""
    collapsed = {}
    pattern_seen = {}
    for param_category in params_dict:
        collapsed[param_category] = {}
        for param_name, param_type in params_dict[param_category].items():
            pattern = r"\.\d+"
            generic_pattern, n = re.subn(pattern, ".{}", param_name)
            if n > 0:
                letters = (letter for letter in "ijklmnopqrstuvwxyzabcdefgh")
                generic_pattern = re.sub(r"\{\}", lambda _: next(letters), generic_pattern)
                if generic_pattern not in pattern_seen:
                    pattern_seen[generic_pattern] = True
                    collapsed[param_category][generic_pattern] = param_type
            else:
                collapsed[param_category][param_name] = param_type
    return collapsed

def match_combo(lst: List[Any] | Tuple[Any]) -> str:
    """Detects comfy dtype for a combo parameter."""
    types_matcher = {
        "str": "STRING", "float": "FLOAT", "int": "INT", "bool": "BOOLEAN"
    }
    if len(lst) > 0:
        return f"{types_matcher.get(type(lst[0]).__name__, 'STRING')}"
    return "STRING"

def get_node_info(node_info: Dict[str, Any]) -> Dict[str, Any]:
    """Collects available information from node class to use in the pipeline."""
    node_class = node_info["class"]
    input_parameters, output_parameters = {}, {}
    for k, node_param_meta in node_class.INPUT_TYPES().items():
        if k in ["required", "optional"]:
            input_parameters[k] = {}
            for param_key, param_meta in node_param_meta.items():
                # skip list
                if param_key in ['tooltips']:
                    continue
                lst = None
                typ = param_meta[0]
                if isinstance(typ, list):
                    typ = match_combo(typ)
                    lst = param_meta
                input_parameters[k][param_key] = {
                    "type": typ
                }
                try:
                    meta = param_meta[1]
                    if lst is not None:
                        if (choice_list := meta.get('choice', None)) is None:
                            data = [x.replace('_', ' ') for x in lst[0]][:JOV_LIST_MAX]
                            input_parameters[k][param_key]["choice"] = data
                            meta.update(lst[1])
                        else:
                            input_parameters[k][param_key]["choice"] = [choice_list][:JOV_LIST_MAX]
                            meta['default'] = 'dynamic'
                    elif (default_top := meta.get('default_top', None)) is not None:
                        meta['default'] = default_top

                    # only stuff that makes sense...
                    junk = ['default', 'min', 'max']
                    meta = node_param_meta[param_key][1]
                    if (tip := meta.get('tooltip', None)) is None:
                        if (tip := Lexicon._tooltipsDB.get(param_key, None)) is None:
                            # logger.warning(f"no tooltip for {node_class}[{k}]::{param_key}")
                            junk.append('tooltip')
                            tip = "Unknown Explanation!"
                    input_parameters[k][param_key]['tooltip'] = tip
                    for scrape in junk:
                        if (val := meta.get(scrape, None)) is not None and val != "":
                            input_parameters[k][param_key][scrape] = val
                except IndexError:
                    pass

    return_types = [
        match_combo(x) if isinstance(x, list) or isinstance(x, tuple) else x for x in node_class.RETURN_TYPES
    ]
    return_names = getattr(node_class, "RETURN_NAMES", [t.lower() for t in return_types])
    for t, n in zip(return_types, return_names):
        output_parameters[n] = ', '.join([x.strip() for x in t.split(',')])
    return {
        "class": repr(node_class).split("'")[1],
        "input_parameters": collapse_repeating_parameters(input_parameters),
        "output_parameters": output_parameters,
        "display_name": node_info["display_name"],
        "output_node": str(getattr(node_class, "OUTPUT_NODE", False)),
        "category": str(getattr(node_class, "CATEGORY", "")),
        "documentation": str(getattr(node_class, "DESCRIPTION", "")),
    }

def json2markdown(json_dict) -> str:
    """Example of json to markdown converter. You are welcome to change formatting per specific request."""
    name = json_dict['display_name']
    boop = name.split('(JOV)')[0].strip()
    boop2 = boop.replace(" ", "%20")
    root1 = f"https://github.com/Amorano/Jovimetrix-examples/blob/master/node/{boop2}/{boop2}.md"
    root2 = f"https://raw.githubusercontent.com/Amorano/Jovimetrix-examples/master/node/{boop2}/{boop2}.png"

    ret = f"## [{name}]({root1})\n\n"
    ret += f"## {json_dict['category']}\n\n"
    ret += f"{json_dict['documentation']}\n\n"
    ret += f"![{boop}]({root2})\n\n"
    ret += f"#### OUTPUT NODE?: `{json_dict['output_node']}`\n\n"

    # INPUTS
    ret += f"## INPUT\n\n"
    if len(json_dict['input_parameters']) > 0:
        for k, v in json_dict['input_parameters'].items():
            if len(v.items()) == 0:
                continue
            ret += f"### {k.upper()}\n\n"
            ret += f"name | type | desc | default | meta\n"
            ret += f":---:|:---:|---|:---:|---\n"
            for param_key, param_meta in v.items():
                typ = param_meta.get('type','UNKNOWN').upper()
                typ = ', '.join([x.strip() for x in typ.split(',')])
                typ = "<br>".join(textwrap.wrap(typ, 42))
                tool = param_meta.get('tooltip','')
                tool = "<br>".join(textwrap.wrap(tool, 42))
                default = param_meta.get('default','')
                ch = ", ".join(param_meta.get('choice', []))
                ch = "<br>".join(textwrap.wrap(ch, 42))
                param_key = param_key.replace('#', r'\#')
                ret += f"{param_key}  |  {typ}  | {tool} | {default} | {ch}\n"
    else:
        ret += 'NONE\n'

    # OUTPUTS
    ret += f"\n## OUTPUT\n\n"
    if len(json_dict['output_parameters']) > 0:
        ret += f"name | type | desc\n"
        ret += f":---:|:---:|---\n"
        for k, v in json_dict['output_parameters'].items():
            if (tool := Lexicon._tooltipsDB.get(k, "")) != "":
                tool = "<br>".join(textwrap.wrap(tool, 65))
            k = k.replace('#', r'\#')
            ret += f"{k}  |  {v}  | {tool} \n"
    else:
        ret += 'NONE\n'

    # BODY INSERT
    # PUT EXTERNAL DOCS HERE
    #
    # FOOTER
    ret += "\noriginal help system powered by [MelMass](https://github.com/melMass) & the [comfy_mtb](https://github.com/melMass/comfy_mtb) project"
    return ret

def json2html(json_dict) -> str:
    """Convert JSON to HTML using templates for all HTML elements."""
    name = json_dict['display_name']
    boop = name.split(' (JOV)')[0].strip()
    boop2 = boop.replace(" ", "%20")
    root1 = f"https://github.com/Amorano/Jovimetrix-examples/blob/master/node/{boop2}/{boop2}.md"
    root2 = f"https://raw.githubusercontent.com/Amorano/Jovimetrix-examples/master/node/{boop2}/{boop2}.png"

    global TEMPLATE
    def template_load(what: str, fname: str) -> Template:
        if TEMPLATE.get(what, None) is None:
            with open(ROOT_DOC / fname, 'r') as f:
                TEMPLATE[what] = Template(f.read())
        return TEMPLATE[what]

    template_node = template_load('node', 'template_node.html')
    input_section = template_load('input_section', 'template_section.html')
    input_row = template_load('input_row', 'template_param_input.html')
    output_row = template_load('output_row', 'template_param_output.html')

    # Generate input content
    input_sections = []
    for k, v in json_dict['input_parameters'].items():
        if not v:
            continue
        rows = []
        for param_key, param_meta in v.items():
            typ = param_meta.get('type', 'UNKNOWN').upper()
            typ = ', '.join([x.strip() for x in typ.split(',')])
            typ = '<br>'.join(textwrap.wrap(typ, 42))
            tool = param_meta.get('tooltip', '')
            tool = '<br>'.join(textwrap.wrap(tool, 42))
            default = html.escape(str(param_meta.get('default', '')))
            ch = ', '.join(param_meta.get('choice', []))
            ch = '<br>'.join(textwrap.wrap(ch, 42))
            rows.append(input_row.substitute(
                param_key=html.escape(param_key),
                type=typ,
                tooltip=tool,
                default=default,
                choice=ch
            ))

        input_sections.append(input_section.substitute(
            name=html.escape(k.upper()),
            rows=''.join(rows)
        ))

    # Generate output content
    output_rows = []
    for k, v in json_dict['output_parameters'].items():
        tool = Lexicon._tooltipsDB.get(k, "")
        tool = '<br>'.join(textwrap.wrap(tool, 65))
        output_rows.append(output_row.substitute(
            name=html.escape(k),
            type=html.escape(v),
            description=tool
        ))

    # Fill in the main template
    html_content = template_node.substitute(
        title=html.escape(name),
        name=html.escape(name),
        root1=root1,
        category=html.escape(json_dict['category']),
        documentation=html.escape(json_dict['documentation']).replace('\n', '<br>'),
        root2=root2,
        boop=html.escape(boop),
        output_node=json_dict['output_node'],
        input_content=''.join(input_sections),
        output_content=''.join(output_rows)
    )
    return html_content

# =============================================================================
# == API RESPONSE
# =============================================================================

class TimedOutException(Exception): pass

class ComfyAPIMessage:
    # STASH = {}
    MESSAGE = {}

    #@classmethod
    #def send(cls, ident, message) -> None:
        #cls.MESSAGE[str(ident)] = message

    @classmethod
    def poll(cls, ident, period=0.01, timeout=3) -> Any:
        _t = time.monotonic()
        if isinstance(ident, (set, list, tuple, )):
            ident = ident[0]
        sid = str(ident)
        while not (sid in cls.MESSAGE) and time.monotonic() - _t < timeout:
            time.sleep(period)

        if not (sid in cls.MESSAGE):
            # logger.warning(f"message failed {sid}")
            raise TimedOutException
        dat = cls.MESSAGE.pop(sid)
        return dat

def comfy_message(ident:str, route:str, data:dict) -> None:
    data['id'] = ident
    PromptServer.instance.send_sync(route, data)

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
        global JOV_CONFIG, JOV_CONFIG_FILE
        if len(JOV_CONFIG) == 0:
            JOV_CONFIG = configLoad(JOV_CONFIG_FILE)
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
        from Jovimetrix.sup.util import update_nested_dict
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

    @PromptServer.instance.routes.get("/jovimetrix/doc")
    async def jovimetrix_doc(request) -> Any:
        data = {}
        for k, v in NODE_CLASS_MAPPINGS.items():
            node = NODE_DISPLAY_NAME_MAPPINGS[k]
            if node.endswith('🧙🏽‍♀️'):
                continue

            if not node.endswith('🧙🏽‍♀️') and DOCUMENTATION.get(node, None) is None:
                ret = {"class": NODE_CLASS_MAPPINGS[node], "display_name": node}
                data = get_node_info(ret)
                DOCUMENTATION[node] = json2html(data)
                data[k]['.html'] = DOCUMENTATION[node]
                data[k]['.md'] = json2markdown(data[k])

                fname = node.split(" (JOV)")[0]
                path = Path(JOV_INTERNAL_DOC.replace("{name}", fname))
                path.mkdir(parents=True, exist_ok=True)

                if not JOV_INTERNAL:
                    with open(str(path / f"{fname}.md"), "w", encoding='utf-8') as f:
                        f.write(data[k]['.md'])

                    with open(str(path / f"{fname}.html"), "w", encoding='utf-8') as f:
                        f.write(html)

        return web.json_response(data)

    @PromptServer.instance.routes.get("/jovimetrix/doc/{node}")
    async def jovimetrix_doc_node(request) -> Any:
        node = request.match_info.get('node')
        docs = f"unknown node: {node}"
        if not node.endswith('🧙🏽‍♀️') and DOCUMENTATION.get(node, None) is None:
            ret = {"class": NODE_CLASS_MAPPINGS[node], "display_name": node}
            data = get_node_info(ret)
            docs = DOCUMENTATION[node] = json2html(data)
        return web.Response(text=docs, content_type='text/html')

except Exception as e:
    logger.error(e)

# =============================================================================
# == SUPPORT FUNCTIONS
# =============================================================================

def parse_reset(ident:str) -> int:
    try:
        data = ComfyAPIMessage.poll(ident, timeout=0)
        return data.get('cmd', None) == 'reset'
    except TimedOutException as e:
        return -1
    except Exception as e:
        logger.error(str(e))

# =============================================================================
# === SESSION ===
# =============================================================================

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

class Session(metaclass=Singleton):
    CLASS_MAPPINGS = {}
    CLASS_MAPPINGS_WIP = {}

    @classmethod
    def ignore_files(cls, d, files) -> list[str]|None:
        return [x for x in files if x.endswith('.json') or x.endswith('.html')]

    def __init__(self, *arg, **kw) -> None:
        global JOV_CONFIG, JOV_IGNORE_NODE, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, NODE_LIST_MAP
        found = False
        if JOV_CONFIG_FILE.exists():
            JOV_CONFIG = configLoad(JOV_CONFIG_FILE)
            # is this an old config, copy default (sorry, not sorry)
            found = JOV_CONFIG.get('user', None) is not None

        if not found:
            try:
                shutil.copy2(JOV_DEFAULT, JOV_CONFIG_FILE)
                logger.warning("---> DEFAULT CONFIGURATION <---")
            except:
                raise Exception("MAJOR 😿😰😬🥟 BLUNDERCATS 🥟😬😰😿")

        if JOV_IGNORE_NODE.exists():
            JOV_IGNORE_NODE = configLoad(JOV_IGNORE_NODE, False)
        else:
            JOV_IGNORE_NODE = []

        node_count = 0
        for f in (ROOT / 'core').iterdir():
            if f.suffix != ".py" or f.stem.startswith('_'):
                continue
            if f.stem in JOV_IGNORE_NODE or f.stem+'.py' in JOV_IGNORE_NODE:
                logger.warning(f"💀 [IGNORED] Jovimetrix.core.{f.stem}")
                continue
            try:
                module = importlib.import_module(f"Jovimetrix.core.{f.stem}")
            except Exception as e:
                logger.warning(f"module failed {f}")
                logger.warning(str(e))
                continue

            # check if there is a dynamic register function....
            try:
                for class_name, class_def in module.import_dynamic():
                    setattr(module, class_name, class_def)
                    logger.info(f"shader: {class_name}")
            except Exception as e:
                pass

            classes = inspect.getmembers(module, inspect.isclass)
            for class_name, class_object in classes:
                # assume both attrs are good enough....
                if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                    if (name := class_object.NAME) in JOV_IGNORE_NODE:
                        logger.warning(f"😥 {name}")
                        continue

                    if hasattr(class_object, 'POST'):
                        class_object.CATEGORY = "JOVIMETRIX 🔺🟩🔵/WIP ☣️💣"
                        Session.CLASS_MAPPINGS_WIP[name] = class_object
                    else:
                        Session.CLASS_MAPPINGS[name] = class_object

                    desc = class_object.DESCRIPTION if hasattr(class_object, 'DESCRIPTION') else ""
                    NODE_LIST_MAP[name] = desc.split('.')[0].strip('\n')
                    node_count += 1

            logger.info(f"✅ {module.__name__}")
        logger.info(f"{node_count} nodes loaded")

        NODE_DISPLAY_NAME_MAPPINGS = {k: v.NAME_PRETTY if hasattr(v, 'NAME_PRETTY') else k for k, v in Session.CLASS_MAPPINGS.items()}
        Session.CLASS_MAPPINGS.update({k: v for k, v in Session.CLASS_MAPPINGS_WIP.items()})
        NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k in Session.CLASS_MAPPINGS_WIP.keys()})
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
                    logger.debug(f"✅ {k} :: {NODE_DISPLAY_NAME_MAPPINGS[k]}")

        # anything we dont know about sort last...
        for k, v in Session.CLASS_MAPPINGS.items():
            NODE_CLASS_MAPPINGS[k] = v
            # logger.debug('⁉️ {} {}', k, v)

        # only do the list on local runs...
        if JOV_INTERNAL:
            with open(str(ROOT) + "/node_list.json", "w", encoding="utf-8") as f:
                json.dump(NODE_LIST_MAP, f, sort_keys=True, indent=4 )

session = Session()
