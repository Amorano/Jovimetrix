"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

              Animation, Image Compositing & Procedural Creation
                    http://www.github.com/amorano/jovimetrix

@title: Jovimetrix
@author: Alexander G. Morano
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
    ConstantNode, GLSLNode, ShapeNode, StereogramNode, StereoscopicNode, TextNode,
    WaveGraphNode,
    AdjustNode, BlendNode, ColorBlindNode, ColorMatchNode, ColorTheoryNode, CropNode,
    FilterMaskNode, FlattenNode, GradientMapNode, PixelMergeNode, PixelSplitNode,
    PixelSwapNode, StackNode, ThresholdNode,TransformNode,
    ComparisonNode, DelayNode, LerpNode, CalcUnaryOPNode, CalcBinaryOPNode,
    StringerNode, SwizzleNode, TickNode, ValueNode, WaveGeneratorNode,
    MIDIFilterNode, MIDIFilterEZNode, MIDIMessageNode, MIDIReaderNode, SpoutWriter,
    StreamReaderNode, StreamWriterNode,
    AkashicNode, ArrayNode, ExportNode, ValueGraphNode, ImageInfoNode, QueueNode,
    QueueTooNode, RouteNode, SaveOutputNode
"""

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__author__ = """Alexander G. Morano"""
__email__ = "amorano@gmail.com"

import os
import re
import sys
import html
import time
import json
import shutil
import inspect
import importlib
from pathlib import Path
from string import Template
from types import ModuleType
from typing import Any, Dict, List, Literal, Tuple

try:
    from markdownify import markdownify
except:
    markdownify = None

from aiohttp import web, ClientSession
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
JOV_DEFAULT = JOV_WEB / 'default.json'
JOV_CONFIG_FILE = JOV_WEB / 'config.json'

# nodes to skip on import; for online systems; skip Export, Streamreader, etc...
JOV_IGNORE_NODE = ROOT / 'ignore.txt'

JOV_LOG_LEVEL = os.getenv("JOV_LOG_LEVEL", "INFO")
logger.configure(handlers=[{"sink": sys.stdout, "level": JOV_LOG_LEVEL}])

JOV_INTERNAL = os.getenv("JOV_INTERNAL", 'false').strip().lower() in ('true', '1', 't')

# direct the documentation output -- used to build jovimetrix-examples
JOV_INTERNAL_DOC = os.getenv("JOV_INTERNAL_DOC", str(ROOT / "_doc"))

JOV_DOCKERENV = os.path.exists('/.dockerenv')

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
    LUT = '😎', "Size of each output lut palette square"
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
    QUEUE = 'Q', "Queue"
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
        name = node_cls.NAME.split(" (JOV)")[0].replace(" ", "-").replace(' GEN', 'GENERATOR')
        sep = "%EF%B8%8F-" if name in MARKDOWN else "-"
        cat = '/'.join(node_cls.CATEGORY.split('/')[1:])
        # WIKI URL
        data = {"_": sep + name, "*": node_cls.NAME, "outputs": {}}
        for cat, entry in node.items():
            if cat not in ['optional', 'required', 'outputs']:
                continue
            for k, v in entry.items():
                widget_data = v[1] if isinstance(v, (tuple, list,)) and len(v) > 1 else {}
                # jovimetrix
                if (tip := widget_data.get("tooltips", None)) is None:
                    # cached
                    if (tip := cls._tooltipsDB.get(k), None) is None:
                        # comfyui standard
                        if (tip := widget_data.get("tooltip", None)) is None:
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
    # instance map for caching
    INSTANCE = {}

    @classmethod
    def VALIDATE_INPUTS(cls, *arg, **kw) -> bool:
        # logger.debug(f'validate -- {arg} {kw}')
        return True

    @classmethod
    def INPUT_TYPES(cls, prompt:bool=False, extra_png:bool=False, dynprompt:bool=False) -> dict:
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

class DynamicInputType(dict):
    """A special class to make flexible nodes that pass data to our python handlers.

    Enables both flexible/dynamic input types or a dynamic number of inputs.

    original sourced from rgthree:
        https://github.com/rgthree/rgthree-comfy/blob/dd534e5384be8cf0c0fa35865afe2126ba75ac55/py/utils.py
    """
    def __init__(self, type: Any) -> None:
        self.type = type

    def __getitem__(self, key: Any) -> Tuple[Any]:
        return (self.type, )

    def __contains__(self, key: Any) -> Literal[True]:
        return True

class AnyType(str):
    """AnyType input wildcard trick taken from pythongossss's:

    https://github.com/pythongosssss/ComfyUI-Custom-Scripts
    """
    def __ne__(self, __value: object) -> bool:
        return False

class DynamicOutputType(tuple):
    """A special class that will return additional "AnyType" strings beyond defined values.

    original sourced from Trung0246:
        https://github.com/Trung0246/ComfyUI-0246/blob/fb16466a82553aebdc4d851a483847c2dc0cb953/utils.py#L51

    """
    def __getitem__(self, index) -> Any:
        if index > len(self) - 1:
            return AnyType("*")
        return super().__getitem__(index)

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

# ==============================================================================
# === DOCUMENTATION SUPPORT
# ==============================================================================

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

def template_load(fname: str) -> Template:
    with open(ROOT_DOC / fname, 'r', encoding='utf-8') as f:
        data = Template(f.read())
    return data

HTML_input_section = template_load('template_section.html')
HTML_input_row = template_load('template_param_input.html')
HTML_output_row = template_load('template_param_output.html')
HTML_template_node = template_load('template_node.html')
HTML_template_node_plain = template_load('template_node_plain.html')

def json2html(json_dict: dict) -> str:
    """Convert JSON to HTML using templates for all HTML elements."""
    name = json_dict['name']
    boop = name.split(' (JOV)')[0].strip()
    root1 = root2 = ""
    template_node = HTML_template_node_plain
    if " (JOV)" in name:
        template_node = HTML_template_node
        boop2 = boop.replace(" ", "%20")
        root1 = f"https://github.com/Amorano/Jovimetrix-examples/blob/master/node/{boop2}/{boop2}.md"
        root2 = f"https://raw.githubusercontent.com/Amorano/Jovimetrix-examples/master/node/{boop2}/{boop2}.png"

    # Generate input content
    input_sections = []
    for k, v in json_dict['input_parameters'].items():
        if not v:
            continue
        rows = []
        for param_key, param_meta in v.items():
            typ = param_meta.get('type', 'UNKNOWN').upper()
            typ = ', '.join([x.strip() for x in typ.split(',')])
            tool = param_meta.get("tooltips", '')
            default = html.escape(str(param_meta.get('default', '')))
            ch = ', '.join(param_meta.get('choice', []))
            rows.append(HTML_input_row.substitute(
                param_key=html.escape(param_key),
                type=typ,
                tooltip=tool,
                default=default,
                choice=ch
            ))

        input_sections.append(HTML_input_section.substitute(
            name=html.escape(k.upper()),
            rows=''.join(rows)
        ))

    # Generate output content
    output_rows = []
    for k, v in json_dict['output_parameters'].items():
        tool = Lexicon._tooltipsDB.get(k, "")
        # tool = '<br>'.join(textwrap.wrap(tool, 65))
        output_rows.append(HTML_output_row.substitute(
            name=html.escape(k),
            type=html.escape(v),
            description=tool
        ))

    # Fill in the main template
    description = json_dict['description']
    #if not "<div>" in description and not "<p>" in description:
        #description = markdown.markdown(description)
        # description = html.escape(description)
    description = description.replace('\n', '<br>').replace(f"('", '').replace(f"')", '')

    html_content = template_node.substitute(
        title=html.escape(name),
        name=html.escape(name),
        root1=root1,
        category=html.escape(json_dict['category']),
        documentation=description,
        root2=root2,
        boop=html.escape(boop),
        output_node=json_dict['output_node'],
        input_content=''.join(input_sections),
        output_content=''.join(output_rows)
    )
    return html_content

def get_node_info(node_data: dict) -> Dict[str, Any]:
    """Transform node object_info route result into .html."""
    input_parameters = {}
    for k, node_param_meta in node_data.get('input', {}).items():
        if not k in ["required", "optional"]:
            continue

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
                if (tip := meta.get("tooltips", None)) is None:
                    if (tip := Lexicon._tooltipsDB.get(param_key, None)) is None:
                        # logger.warning(f"no tooltip for {node_class}[{k}]::{param_key}")
                        junk.append("tooltips")
                        tip = "Unknown Explanation!"
                input_parameters[k][param_key]["tooltips"] = tip
                for scrape in junk:
                    if (val := meta.get(scrape, None)) is not None and val != "":
                        input_parameters[k][param_key][scrape] = val
            except IndexError:
                pass

    return_types = [
        match_combo(x) if isinstance(x, list) or isinstance(x, tuple) else x for x in node_data.get('output', [])
    ]

    output_parameters = {}
    return_names = [t.lower() for t in node_data.get('output_name', [])]
    for t, n in zip(return_types, return_names):
        output_parameters[n] = ', '.join([x.strip() for x in t.split(',')])

    data = {
        "class": node_data['name'],
        "input_parameters": collapse_repeating_parameters(input_parameters),
        "output_parameters": output_parameters,
        "name": node_data['name'],
        "output_node": node_data['output_node'],
        "category": node_data['category'].strip('\n').strip(),
        "description": node_data['description']
    }
    data[".html"] = json2html(data)
    if markdownify:
        md = markdownify(data[".html"], keep_inline_images_in=True)
        md = md.split('\n')[8:]
        md = '\n'.join([m for m in md if m != ''])
        data[".md"] = md
    return data

def deep_merge(d1: dict, d2: dict) -> dict:
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
        _t = time.monotonic()
        if isinstance(ident, (set, list, tuple, )):
            ident = ident[0]
        sid = str(ident)
        logger.debug(f'sid {sid} -- {cls.MESSAGE}')
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

    @PromptServer.instance.routes.get("/jovimetrix")
    async def jovimetrix_home(request) -> Any:
        data = template_load('home.html')
        return web.Response(text=data.template, content_type='text/html')

    @PromptServer.instance.routes.get("/jovimetrix/message")
    async def jovimetrix_message(request) -> Any:
        return web.json_response(ComfyAPIMessage.MESSAGE)

    @PromptServer.instance.routes.post("/jovimetrix/message")
    async def jovimetrix_message_post(request) -> Any:
        json_data = await request.json()
        logger.info(json_data)
        if (did := json_data.get("id")) is not None:
            ComfyAPIMessage.MESSAGE[str(did)] = json_data
            return web.json_response(json_data)
        return web.json_response({})

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
        value = json_data.get("cmd", None)
        if did is None or value is None:
            logger.error("bad config {}", json_data)
            return

        global JOV_CONFIG
        update_nested_dict(JOV_CONFIG, did, value)
        with open(JOV_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(JOV_CONFIG, f, indent=4)
        return web.json_response(json_data)

    @PromptServer.instance.routes.post("/jovimetrix/config/clear")
    async def jovimetrix_config_post(request) -> Any:
        json_data = await request.json()
        name = json_data['name']
        global JOV_CONFIG
        try:
            del JOV_CONFIG['color'][name]
        except KeyError as _:
            pass
        with open(JOV_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(JOV_CONFIG, f)
        return web.json_response(json_data)

    async def object_info(node_class: str, scheme:str, host: str) -> Any:
        global COMFYUI_OBJ_DATA
        if (info := COMFYUI_OBJ_DATA.get(node_class, None)) is None:
            # look up via the route...
            url = f"{scheme}://{host}/object_info/{node_class}"

            # Make an asynchronous HTTP request using aiohttp.ClientSession
            async with ClientSession() as session:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            info = await response.json()
                            if (data := info.get(node_class, None)) is not None:
                                info = get_node_info(data)
                            else:
                                info = {'.html': f"No data for {node_class}"}
                            COMFYUI_OBJ_DATA[node_class] = info
                        else:
                            info = {'.html': f"Failed to get docs {node_class}, status: {response.status}"}
                            logger.error(info)
                except Exception as e:
                    logger.error(f"Failed to get docs {node_class}")
                    logger.exception(e)
                    info = {'.html': f"Failed to get docs {node_class}\n{e}"}

        return info

    @PromptServer.instance.routes.get("/jovimetrix/doc")
    async def jovimetrix_doc(request) -> Any:

        for node_class in NODE_CLASS_MAPPINGS.keys():
            if COMFYUI_OBJ_DATA.get(node_class, None) is None:
                COMFYUI_OBJ_DATA[node_class] = await object_info(node_class, request.scheme, request.host)

            node = NODE_DISPLAY_NAME_MAPPINGS[node_class]
            fname = node.split(" (JOV)")[0]
            path = Path(JOV_INTERNAL_DOC.replace("{name}", fname))
            path.mkdir(parents=True, exist_ok=True)

            if JOV_INTERNAL:
                if (md := COMFYUI_OBJ_DATA[node_class].get('.md', None)) is not None:
                    with open(str(path / f"{fname}.md"), "w", encoding='utf-8') as f:
                        f.write(md)

                with open(str(path / f"{fname}.html"), "w", encoding='utf-8') as f:
                    f.write(COMFYUI_OBJ_DATA[node_class]['.html'])

        return web.json_response(COMFYUI_OBJ_DATA)

    @PromptServer.instance.routes.get("/jovimetrix/doc/{node}")
    async def jovimetrix_doc_node_comfy(request) -> Any:
        node_class = request.match_info.get('node')
        if COMFYUI_OBJ_DATA.get(node_class, None) is None:
            COMFYUI_OBJ_DATA[node_class] = await object_info(node_class, request.scheme, request.host)
        return web.Response(text=COMFYUI_OBJ_DATA[node_class]['.html'], content_type='text/html')

except Exception as e:
    logger.error(e)

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
    try:
        route = str(name).replace("\\", "/")
        route = route.split(f"{module}/core/")[1]
        route = route.split('.')[0].replace('/', '.')
    except Exception as e:
        logger.warning(f"module failed {name}")
        logger.warning(str(e))
        return

    try:
        module = f"{module}.core.{route}"
        module = importlib.import_module(module)
    except Exception as e:
        logger.warning(f"module failed {module}")
        logger.warning(str(e))
        return

    return module

def loader():
    global JOV_CONFIG, JOV_IGNORE_NODE, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    NODE_LIST_MAP = {}

    found = False
    if JOV_CONFIG_FILE.exists():
        JOV_CONFIG = configLoad(JOV_CONFIG_FILE)
        # is this an old config, copy default (sorry, not sorry)
        found = JOV_CONFIG.get('user', None) is not None

    if not found:
        try:
            shutil.copy2(JOV_DEFAULT, JOV_CONFIG_FILE)
            logger.warning("---> DEFAULT CONFIGURATION <---")
        except Exception as e:
            logger.error("MAJOR 😿😰😬🥟 BLUNDERCATS 🥟😬😰😿")
            logger.error(e)

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
    for name in keys:
        logger.debug(f"✅ {name}")
    logger.info(f"{len(keys)} nodes loaded")

    # only do the list on local runs...
    if JOV_INTERNAL:
        with open(str(ROOT) + "/node_list.json", "w", encoding="utf-8") as f:
            json.dump(NODE_LIST_MAP, f, sort_keys=True, indent=4 )

# ==============================================================================
# === BOOTSTRAP ===
# ==============================================================================

loader()
