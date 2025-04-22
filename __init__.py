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

__author__ = "Alexander G. Morano"
__email__ = "amorano@gmail.com"

from pathlib import Path
from typing import Any, Dict

from cozy_comfyui import \
    logger

from cozy_comfyui.node import \
    loader

JOV_DOCKERENV = False
try:
    with open('/proc/1/cgroup', 'rt') as f:
        content = f.read()
        JOV_DOCKERENV = any(x in content for x in ['docker', 'kubepods', 'containerd'])
except FileNotFoundError:
    pass

if JOV_DOCKERENV:
    logger.info("RUNNING IN A DOCKER")

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

# HSV = 'HSV', "Hue, Saturation and Value"
# INDEX = 'INDEX', "Current item index in the Queue list"
# JUSTIFY = 'JUSTIFY', "How to align the text to the side margins of the canvas: Left, Right, or Centered"
# LETTER = 'LETTER', "If each letter be generated and output in a batch"
# LMH = 'LMH', "Low, Middle, High"
# MARGIN = 'MARGIN', "Whitespace padding around canvas"
# MODE = 'MODE', "Decide whether the images should be resized to fit a specific dimension. Available modes include scaling to fit within given dimensions or keeping the original size"
# QUALITY_M = 'MOTION', "Motion Quality"
# RANGE = 'RANGE', "start index, ending index (0 means full length) and how many items to skip per step"
# REPLACE = 'REPLACE', "String to use as replacement"
# SHAPE = 'SHAPE', "Circle, Square or Polygonal forms"
# SIDES = 'SIDES', "Number of sides polygon has (3-100)"
# SPACING = 'SPACING', "Line Spacing between Text Lines"
# START = 'START', "Start of the range"
# TILE = 'TILE', "How many times to repeat the data in the X and Y"
# TOTAL = 'TOTAL', "Total items in the current Queue List"

class Lexicon(metaclass=LexiconMeta):
    A = '⬜', "Alpha"
    AMP = '🔊', "Amplitude"
    ANGLE = '📐', "Rotation Angle"
    ANY = '🔮', "Any Type"
    ANY_OUT = '🦄', "Any Type"
    B = '🟦', "Blue"
    BI = '💙', "Blue Channel"
    BOOLEAN = '🇴', "Boolean"
    BOTTOM = '🔽', "Bottom"
    C1 = '🔵', "Color Scheme Result 1"
    C2 = '🟡', "Color Scheme Result 2"
    C3 = '🟣', "Color Scheme Result 3"
    C4 = '⚫️', "Color Scheme Result 4"
    C5 = '⚪', "Color Scheme Result 5"
    COLORMAP = '🇸🇨', "One of two dozen CV2 Built-in Colormap LUT (Look Up Table) Presets"
    COMP_A = '😍', "pass this data on a successful condition"
    COMP_B = '🥵', "pass this data on a failure condition"
    COMPARE = '🕵🏽‍♀️', "Comparison function. Will pass the data in 😍 on successful comparison"
    CONTRAST = '🌓', "Contrast"
    FLIP = '🙃', "Flip Input A and Input B with each other"
    FLOAT = '🛟', "Float"
    FOCAL = '📽️', "Focal Length"
    FPS = '🏎️', "Frames per second"
    FUNC = '⚒️', "Function"
    G = '🟩', "Green"
    GAMMA = '🔆', "Gamma"
    GI = '💚', "Green Channel"
    GRADIENT = '🇲🇺', "Gradient"
    H = '🇭', "Hue"
    IMAGE = '🖼️', "RGB-A color image with alpha channel"
    IN_A = '🅰️', "Input A"
    IN_B = '🅱️', "Input B"
    INT = '🔟', "Integer"
    INVERT = '🔳', "Color Inversion"
    KEY = '🔑', "Key"
    LEFT = '◀️', "Left"
    LINEAR = '🛟', "Linear"
    LIST = '🧾', "List"
    LOOP = '🔄', "Loop"
    LUT = '😎', "Size of each output lut palette square"
    MASK = '😷', "Mask or Image to use as Mask to control where adjustments are applied"
    MI = '🤍', "Alpha Channel"
    MIRROR = '🪞', "Mirror"
    NOTE = '🎶', "Note"
    PALETTE = '🎨', "Palette"
    PASS_IN = '📥', "Pass In"
    PASS_OUT = '📤', "Pass Out"
    PIXEL = '👾', "Pixel Data (RGBA, RGB or Grayscale)"
    PIXEL_A = '👾A', "Pixel Data (RGBA, RGB or Grayscale)"
    PIXEL_B = '👾B', "Pixel Data (RGBA, RGB or Grayscale)"
    R = '🟥', "Red"
    RADIUS = '🅡', "Radius"
    RGB = '🌈', "RGB (no alpha) Color"
    RGB_A = '🌈A', "RGB (no alpha) Color"
    RGBA_A = '🌈A', "RGB with Alpha Color"
    RI = '❤️', "Red Channel"
    RIGHT = '▶️', "Right"
    ROUTE = '🚌', "Route"
    S = '🇸', "Saturation"
    SAMPLE = '🎞️', "Method for resizing images."
    SIZE = '📏', "Scalar by which to scale the input"
    STEP = '🦶🏽', "Steps/Stride between pulses -- useful to do odd or even batches. If set to 0 will stretch from (VAL -> LOOP) / Batch giving a linear range of values."
    STRENGTH = '💪🏽', "Strength"
    STRING = '📝', "String Entry"
    THRESHOLD = '📉', "Threshold"
    TIME = '🕛', "Time"
    TIMER = '⏱', "Timer"
    TOP = '🔼', "Top"
    TRIGGER = '⚡', "Trigger"
    TYPE = '❓', "Type"
    UNKNOWN = '❔', "Unknown"
    V = '🇻', "Value"
    W = '🇼', "Width"
    WAIT = '✋🏽', "Wait"
    WAVE = '♒', "Wave Function"
    WH = '🇼🇭', "Width and Height as a Vector2 (x,y)"
    WHC = '🇼🇭🇨', "Width, Height and Channel as a Vector3 (x,y,z)"
    X = '🇽', "X"
    XY = '🇽🇾', "X and Y"
    Y = '🇾', "Y"
    Z = '🇿', "Z"

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
# === GLOBAL ===
# ==============================================================================

PACKAGE = "JOVIMETRIX"
WEB_DIRECTORY = "./web"
ROOT = Path(__file__).resolve().parent
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = loader(ROOT,
                                                         PACKAGE,
                                                         "core",
                                                         f"{PACKAGE} 🔺🟩🔵",
                                                         False)
