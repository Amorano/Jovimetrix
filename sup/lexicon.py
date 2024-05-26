"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
EMOJI OCD Support
"""

# 🔗 ⚓ 📀 🧹 🍿 ➕ 📽️ 🦄 📑 📺 🎪 🐘 🚦 🤯 😱 💀 ⛓️ 🔒 🪀 🪁 🧿 🧙🏽 🧙🏽‍♀️
# 🧯 🦚 ♻️  ⤴️ ⚜️ 🅱️ 🅾️ ⬆️ ↔️ ↕️ 〰️ ☐ 🚮 🤲🏽 👍 ✳️ ✌🏽 ☝🏽

import re
import textwrap
from typing import Any, Dict, List, Tuple
from loguru import logger

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
    API = 'API', "API URL route"
    ATTRIBUTE = 'ATTRIBUTE', "The token attribute to use for authenticating"
    AUTH = 'AUTH', "Authentication Bearer Token"
    AUTOSIZE = 'AUTOSIZE', "Scale based on Width & Height"
    AXIS = 'AXIS', "Axis"
    B = '🟦', "Blue"
    BATCH = 'BATCH', "Process multiple images"
    BATCH_CHUNK = 'CHUNK', "How many items to put per output. Default (0) is all items"
    BATCH_LIST = 'AS LIST', "Process each entry as a list"
    BATCH_MODE = 'MODE', "Make, merge, splice or split a batch or list"
    BATCH_SELECT = 'SELECT', "How to pick items from the list -- by index or randomly"
    BBOX = '🔲', "Bounding box"
    BEAT = '🥁', "Beats per minute"
    BI = '💙', "Blue Channel"
    BLACK = '⬛', "Black Channel"
    BLBR = 'BL-BR', "Bottom Left - Bottom Right"
    BLUR = 'BLUR', "Blur"
    BOOLEAN = '🇴', "Boolean"
    BOTTOM = '🔽', "Bottom"
    BPM = 'BPM', "The number of Beats Per Minute"
    C1 = '🔵', "Color Scheme 1"
    C2 = '🟡', "Color Scheme 2"
    C3 = '🟣', "Color Scheme 3"
    C4 = '⚫️', "Color Scheme 4"
    C5 = '⚪', "Color Scheme 5"
    CAMERA = '📹', "Camera"
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
    CURRENT = 'CURRENT', "Current"
    DATA = '📓', "Data"
    DEFIENCY = 'DEFIENCY', "The type of color blindness: Red-Blind/Protanopia, Green-Blind/Deuteranopia or Blue-Blind/Tritanopia"
    DELAY = '✋🏽', "Delay"
    DELTA = '🔺', "Delta"
    DEPTH = 'DEPTH', "Grayscale image representing a depth map"
    DEVICE = '📟', "Device"
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
    FOLDER = '📁', "Folder"
    FONT = 'FONT', "Available System Fonts"
    FONT_SIZE = 'SIZE', "Text Size"
    FORMAT = 'FORMAT', "Format"
    FPS = '🏎️', "Frames per second"
    FRAGMENT = 'FRAGMENT', "Shader Fragment Program"
    FRAME = '⏹️', "Frame"
    FREQ = 'FREQ', "Frequency"
    FUNC = '⚒️', "Function"
    G = '🟩', "Green"
    GAMMA = '🔆', "Gamma"
    GI = '💚', "Green Channel"
    GRADIENT = '🇲🇺', "Gradient"
    H = '🇭', "Hue"
    HI = 'HI', "High / Top of range"
    HSV = u'🇭🇸\u200c🇻', "Hue, Saturation and Value"
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
    MATTE = 'MATTE', "Background Color"
    MAX = 'MAX', "Maximum"
    MI = '🤍', "Alpha Channel"
    MID = 'MID', "Middle"
    MIDI = '🎛️', "Midi"
    MIRROR = '🪞', "Mirror"
    MODE = 'MODE', "Scaling Mode"
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
    PROJECTION = 'PROJ', "Projection"
    QUALITY = 'QUALITY', "Quality"
    QUALITY_M = 'MOTION', "Motion Quality"
    QUEUE = 'Q', "Queue"
    R = '🟥', "Red"
    RADIUS = '🅡', "Radius"
    RANDOM = 'RNG', "Random"
    RANGE = 'RANGE', "start index, ending index (0 means full length) and how many items to skip per step"
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
    SAMPLE = '🎞️', "Sampling Method to apply when Rescaling"
    SCHEME = 'SCHEME', "Scheme"
    SEED = 'SEED', "Seed"
    SEGMENT = 'SEGMENT', "Number of parts which the input image should be split"
    SELECT = 'SELECT', "Select"
    SHAPE = '🇸🇴', "Circle, Square or Polygonal forms"
    SHIFT = 'SHIFT', "Shift"
    SIDES = '♾️', "Number of sides polygon has (3-100)"
    SIMULATOR = 'SIMULATOR', "The solver to use when translating color space"
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
    TOP = '🔼', "Top"
    TOTAL = 'TOTAL', "Total items in the current Queue List"
    TRIGGER = '⚡', "Trigger"
    TRUE = '🇹', "True"
    TYPE = '❓', "Type"
    UNKNOWN = '❔', "Unknown"
    URL = '🌐', "URL"
    V = '🇻', "Value"
    VALUE = 'VALUE', "Value"
    VEC = 'VECTOR', "Compound value of type float, vec2, vec3 or vec4"
    W = '🇼', "Width"
    WAIT = '✋🏽', "Wait"
    WAVE = '♒', "Wave Function"
    WH = '🇼🇭', "Width and Height"
    WINDOW = '🪟', "Window"
    X = '🇽', "X"
    XY = '🇽🇾', "X and Y"
    XYZ = '🇽🇾\u200c🇿', "X, Y and Z (VEC3)"
    XYZW = '🇽🇾\u200c🇿\u200c🇼', "X, Y, Z and W (VEC4)"
    Y = '🇾', "Y"
    Z = '🇿', "Z"
    ZOOM = '🔎', "ZOOM"

    @classmethod
    def _parse(cls, node: dict, node_cls: object) -> dict:
        name_url = node_cls.NAME.split(" (JOV)")[0]
        url = name_url.replace(" ", "%20")
        cat = node_cls.CATEGORY.split('/')[1]
        data = {"_": f"{cat}#-{url}", "*": f"node/{name_url}/{name_url}.md"}
        for cat, entry in node.items():
            if cat not in ['optional', 'required']:
                continue
            for k, v in entry.items():
                if (tip := v[1].get('tooltip', None)) is None:
                    if (tip := cls._tooltipsDB.get(k), None) is None:
                        logger.warning(f"no {k}")
                        continue
                data[k] = tip
        node["optional"]["tooltips"] = ("JTOOLTIP", {"default": data})
        return node

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

def match_combo(lst: List[Any] | Tuple[Any]):
    """Detects comfy dtype for a combo parameter."""
    types_matcher = {
        "str": "STRING", "float": "FLOAT", "int": "INT", "bool": "BOOLEAN"
    }
    if len(lst) > 0:
        return f"COMBO[{types_matcher.get(type(lst[0]).__name__, 'STRING')}]"
    return "COMBO[STRING]"

def get_node_info(node_info: Dict[str, Any]) -> Dict[str, Any]:
    """Collects available information from node class to use in the pipeline."""
    node_class = node_info["class"]
    input_parameters, output_parameters = {}, {}
    for k, v in node_class.INPUT_TYPES().items():
        if k in ["required", "optional"]:
            input_parameters[k] = {}
            for k0, v0 in v.items():
                # skip list
                if k0 in ['tooltips']:
                    continue
                lst = None
                typ = v0[0]
                if isinstance(typ, list):
                    typ = match_combo(typ)
                    lst = v0
                input_parameters[k][k0] = {
                    "type": typ
                }
                meta = v0[1]
                if lst is not None:
                    input_parameters[k][k0]["choice"] = lst[0]
                    meta.update(lst[1])
                # only stuff that makes sense...
                junk = ['default', 'min', 'max']
                if (val := Lexicon._tooltipsDB.get(k0, None)) is not None:
                    input_parameters[k][k0]['tooltip'] = val
                else:
                    junk.append('tooltip')
                for scrape in junk:
                    if (val := meta.get(scrape, None)) is not None and val != "":
                        input_parameters[k][k0][scrape] = val

    return_types = [
        match_combo(x) if isinstance(x, list) or isinstance(x, tuple) else x for x in node_class.RETURN_TYPES
    ]
    return_names = getattr(node_class, "RETURN_NAMES", [t.lower() for t in return_types])
    for t, n in zip(return_types, return_names):
        output_parameters[n] = t
    return {
        "class": repr(node_class).split("'")[1],
        "input_parameters": collapse_repeating_parameters(input_parameters),
        "output_parameters": output_parameters,
        "display_name": node_info["display_name"],
        "output_node": str(getattr(node_class, "OUTPUT_NODE", False)),
        "category": str(getattr(node_class, "CATEGORY", "")),
        "documentation": str(getattr(node_class, "DESCRIPTION", "")),
    }

def json2markdown(json_dict):
    """Example of json to markdown converter. You are welcome to change formatting per specific request."""
    name = json_dict['display_name']
    ret = f"# {name}\n\n"
    ret += f"## {json_dict['category']}\n"
    ret += f"{json_dict['documentation']}\n"
    #name = name.split(" (JOV)")[0].replace(" ", "%20")
    # ret += f"![](https://raw.githubusercontent.com/Amorano/Jovimetrix-examples/master/node/{name}/{name}.gif)\n\n"
    ret += f"#### OUTPUT NODE?: `{json_dict['output_node']}`\n\n"
    ret += f"### INPUT\n\n"
    if len(json_dict['output_parameters']) > 0:
        for k, v in json_dict['input_parameters'].items():
            if len(v.items()) == 0:
                continue
            ret += f"#### {k.upper()}\n\n"
            ret += f"name|type|desc|default|meta\n"
            ret += f":---:|:---:|---|:---:|---\n"
            for param_key, param_meta in v.items():
                typ = param_meta.get('type','UNKNOWN').upper()
                tool = param_meta.get('tooltip','').lower()
                tool = "<br>".join(textwrap.wrap(tool, 35))
                default = param_meta.get('default','')
                ch = ", ".join(param_meta.get('choice', []))
                ch = "<br>".join(textwrap.wrap(ch, 45))
                ret += f"{param_key}| {typ} | {tool} | {default} | {ch}\n"
    else:
        ret += 'NONE\n'
    ret += f"\n### OUTPUT\n\n"
    if len(json_dict['output_parameters']) > 0:
        ret += f"name|type|desc\n"
        ret += f":---:|:---:|---\n"
        for k, v in json_dict['output_parameters'].items():
            if (tool := Lexicon._tooltipsDB.get(k, "")) != "":
                tool = "<br>".join(textwrap.wrap(tool, 40))
            ret += f"{k}| {v} | {tool} \n"
    else:
        ret += 'NONE\n'
    ret += "\nhelp powered by [MelMass](https://github.com/melMass) & [comfy_mtb](https://github.com/melMass/comfy_mtb) project"
    return ret
