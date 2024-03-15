"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
EMOJI OCD Support
"""

# 🔗 ⚓ 📀 🧹 🍿 ➕ 📽️ 🦄 📑 📺 🎪 🐘 🚦 🤯 😱 💀 ⛓️ 🔒 🪀 🪁 🧿 🧙🏽 🧙🏽‍♀️
# 🧯 🦚 ♻️  ⤴️ ⚜️ 🅱️ 🅾️ ⬆️ ↔️ ↕️ 〰️ ☐ 🚮 🤲🏽 👍 ✳️ ✌🏽 ☝🏽

from typing import Any
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
    ADAPT = '🧬', "X-Men"
    ALIGN = 'ALIGN', "Top, Center or Bottom alignment"
    AMP = '🔊', "Amplitude"
    ANGLE = '📐', "Rotation Angle"
    ANY = '🔮', "Any Type"
    AUTOSIZE = 'AUTOSIZE', "Scale based on Width & Height"
    AXIS = 'AXIS', "Axis"
    B = '🟦', "Blue"
    BATCH = 'BATCH', "Process multiple images"
    BATCH_LIST = 'AS LIST', "Process each entry as a list"
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
    END = 'END', "End"
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
    MASK = '😷', "Mask or Image to use as Mask"
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
    RECORD = '⏺', "Arm record capture from selected device"
    REGION = 'REGION', "Region"
    RESET = 'RESET', "Reset"
    RGB = '🌈', "RGB (no alpha) Color"
    RGB_A = '🌈A', "RGB (no alpha) Color"
    RGBA_A = '🌈A', "RGB with Alpha Color"
    RGBA_B = '🌈B', "RGB with Alpha Color"
    RI = '❤️', "Red Channel"
    RIGHT = '▶️', "Right"
    ROUND = 'ROUND', "Round to the nearest decimal place, or 0 for integer mode"
    ROUTE = '🚌', "Route"
    S = '🇸', "Saturation"
    SAMPLE = '🎞️', "Sampling Method to apply when Rescaling"
    SCHEME = 'SCHEME', "Scheme"
    SEED = 'SEED', "Seed"
    SELECT = 'SELECT', "Select"
    SHAPE = '🇸🇴', "Circle, Square or Polygonal forms"
    SHIFT = 'SHIFT', "Shift"
    SIDES = '♾️', "Number of sides polygon has (3-100)"
    SIMULATOR = 'SIMULATOR', "The solver to use when translating color space"
    SIZE = '📏', "Scale"
    SOURCE = 'SRC', "Source"
    SPACING = 'SPACING', "Line Spacing between Text Lines"
    START = 'START', "Start"
    STEP = '🦶🏽', "Step"
    STRENGTH = '💪🏽', "Strength"
    STRING = '📝', "String Entry"
    STYLE = 'STYLE', "Style"
    SWAP_A = 'SWAP A', "Replace input Alpha channel with target channel or solid"
    SWAP_B = 'SWAP B', "Replace input Blue channel with target channel or solid"
    SWAP_G = 'SWAP G', "Replace input Green channel with target channel or solid"
    SWAP_R = 'SWAP R', "Replace input Red channel with target channel or solid"
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
    VALUE = '#️⃣', "Value"
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
    def _parse(cls, node: dict, url: str=None) -> dict:
        data = {}
        if url is not None:
            data["_"] = url

        # the node defines...
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
