"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Calculation
"""

import math
from enum import Enum
from typing import Any
from collections import Counter

import numpy as np
from scipy.special import gamma
from loguru import logger
import torch

from comfy.utils import ProgressBar

from Jovimetrix import JOV_HELP_URL, JOVBaseNode, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import EnumTupleType, parse_tuple, zip_longest_fill
from Jovimetrix.sup.anim import ease_op, EnumEase

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/CALC"

# =============================================================================

class EnumNumberType(Enum):
    INT = 0
    FLOAT = 10

class EnumConvertType(Enum):
    STRING = 0
    BOOLEAN = 10
    INT = 12
    FLOAT = 14
    VEC2 = 20
    VEC2INT = 25
    VEC3 = 30
    VEC3INT = 35
    VEC4 = 40
    VEC4INT = 45

class EnumUnaryOperation(Enum):
    ABS = 0
    FLOOR = 1
    CEIL = 2
    SQRT = 3
    SQUARE = 4
    LOG = 5
    LOG10 = 6
    SIN = 7
    COS = 8
    TAN = 9
    NEGATE = 10
    RECIPROCAL = 12
    FACTORIAL = 14
    EXP = 16
    # COMPOUND
    MINIMUM = 20
    MAXIMUM = 21
    MEAN = 22
    MEDIAN = 24
    MODE = 26
    MAGNITUDE = 30
    NORMALIZE = 32
    # LOGICAL
    NOT = 40
    # BITWISE
    BIT_NOT = 45
    COS_H = 60
    SIN_H = 62
    TAN_H = 64
    RADIANS = 70
    DEGREES = 72
    GAMMA = 80
    # IS_EVEN
    IS_EVEN = 90
    IS_ODD = 91

class EnumBinaryOperation(Enum):
    ADD = 0
    SUBTRACT = 1
    MULTIPLY   = 2
    DIVIDE = 3
    DIVIDE_FLOOR = 4
    MODULUS = 5
    POWER = 6
    # TERNARY WITHOUT THE NEED
    MAXIMUM = 20
    MINIMUM = 21
    # VECTOR
    DOT_PRODUCT = 30
    CROSS_PRODUCT = 31
    # MATRIX

    # BITS
    # BIT_NOT = 39
    BIT_AND = 60
    BIT_NAND = 61
    BIT_OR = 62
    BIT_NOR = 63
    BIT_XOR = 64
    BIT_XNOR = 65
    BIT_LSHIFT = 66
    BIT_RSHIFT = 67
    # GROUP
    UNION = 80
    INTERSECTION = 81
    DIFFERENCE = 82

# =============================================================================

# Dictionary to map each operation to its corresponding function
OP_UNARY = {
    EnumUnaryOperation.ABS: lambda x: math.fabs(x),
    EnumUnaryOperation.FLOOR: lambda x: math.floor(x),
    EnumUnaryOperation.CEIL: lambda x: math.ceil(x),
    EnumUnaryOperation.SQRT: lambda x: math.sqrt(x),
    EnumUnaryOperation.SQUARE: lambda x: math.pow(x, 2),
    EnumUnaryOperation.LOG: lambda x: math.log(x),
    EnumUnaryOperation.LOG10: lambda x: math.log10(x),
    EnumUnaryOperation.SIN: lambda x: math.sin(x),
    EnumUnaryOperation.COS: lambda x: math.cos(x),
    EnumUnaryOperation.TAN: lambda x: math.tan(x),
    EnumUnaryOperation.NEGATE: lambda x: -x,
    EnumUnaryOperation.RECIPROCAL: lambda x: 1 / x,
    EnumUnaryOperation.FACTORIAL: lambda x: math.factorial(int(x)),
    EnumUnaryOperation.EXP: lambda x: math.exp(x),
    EnumUnaryOperation.NOT: lambda x: not x,
    EnumUnaryOperation.BIT_NOT: lambda x: ~int(x),
    EnumUnaryOperation.IS_EVEN: lambda x: x % 2 == 0,
    EnumUnaryOperation.IS_ODD: lambda x: x % 2 == 1,
    EnumUnaryOperation.COS_H: lambda x: math.cosh(x),
    EnumUnaryOperation.SIN_H: lambda x: math.sinh(x),
    EnumUnaryOperation.TAN_H: lambda x: math.tanh(x),
    EnumUnaryOperation.RADIANS: lambda x: math.radians(x),
    EnumUnaryOperation.DEGREES: lambda x: math.degrees(x),
    EnumUnaryOperation.GAMMA: lambda x: gamma(x)
}

# =============================================================================

def parse_type_value(typ:EnumConvertType, A:Any, default:tuple=(0,0,0,0)) -> tuple[EnumConvertType, Any, int]:
    size = max(1, int(typ.value / 10))
    if A is None:
        val = default
    elif isinstance(A, (torch.Tensor,)):
        val = list(A.size())
        val = val[1:4] + [val[0]]
    else:
        val = A
    if not isinstance(val, (list, tuple, set,)):
        val = [val] * size
    return typ, val, size

def convert_value(typ:EnumConvertType, val:Any, size:int) -> Any:
    if typ == EnumConvertType.STRING:
        val = ", ".join([str(v) for v in val])
    elif typ == EnumConvertType.BOOLEAN:
        val = bool(val[0])
    elif typ in [EnumConvertType.FLOAT, EnumConvertType.VEC2,
                    EnumConvertType.VEC3, EnumConvertType.VEC4]:
        val = [round(float(v), 12) for v in val][:size]
    else:
        val = [int(v) for v in val][:size]
    if typ in [EnumConvertType.FLOAT, EnumConvertType.INT]:
        val = val[0]
    return val

# =============================================================================

class CalcUnaryOPNode(JOVBaseNode):
    NAME = "CALC OP UNARY (JOV) 🎲"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Perform a Unary Operation on an input."
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN, )
    OUTPUT_IS_LIST = (True, )
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.FUNC: (EnumUnaryOperation._member_names_, {"default": EnumUnaryOperation.ABS.name})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#-calc-op-unary")

    def run(self, **kw) -> tuple[bool]:
        results = []
        A = kw.get(Lexicon.IN_A, [0])
        op = kw.get(Lexicon.FUNC, [EnumUnaryOperation.ABS])
        params = [tuple(x) for x in zip_longest_fill(A, op)]
        pbar = ProgressBar(len(params))
        for idx, (A, op) in enumerate(params):
            _, val, size = parse_type_value(EnumConvertType.VEC4, A)
            convert = int if isinstance(A, (bool, int, np.uint8, np.uint16, np.uint32, np.uint64)) else float
            val = [convert(v) for v in val][:size]
            op = EnumUnaryOperation[op]
            match op:
                case EnumUnaryOperation.MEAN:
                    val = [sum(val) / len(val)]
                case EnumUnaryOperation.MEDIAN:
                    val = [sorted(val)[len(val) // 2]]
                case EnumUnaryOperation.MODE:
                    counts = Counter(val)
                    val = [max(counts, key=counts.get)]
                case EnumUnaryOperation.MAGNITUDE:
                    val = [math.sqrt(sum(x ** 2 for x in val))]
                case EnumUnaryOperation.NORMALIZE:
                    if len(val) == 1:
                        val = [1]
                    else:
                        m = math.sqrt(sum(x ** 2 for x in val))
                        val = [v / m for v in val]
                case EnumUnaryOperation.MAXIMUM:
                    val = [max(val)]
                case EnumUnaryOperation.MINIMUM:
                    val = [min(val)]
                case _:
                    # Apply unary operation to each item in the list
                    ret = []
                    for v in val:
                        try:
                            v = OP_UNARY[op](v)
                        except Exception as e:
                            logger.error(str(e))
                            v = 0
                        ret.append(v)
                    val = ret
            results.append(val)
            pbar.update_absolute(idx)
        return list(zip(*results))

class CalcBinaryOPNode(JOVBaseNode):
    NAME = "CALC OP BINARY (JOV) 🌟"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Perform a Binary Operation on two inputs."
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN, )
    OUTPUT_IS_LIST = (True, )
    SORT = 20

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        names_convert = EnumConvertType._member_names_[1:]
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None,
                                      "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections."}),
            Lexicon.IN_B: (WILDCARD, {"default": None,
                                      "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections."}),
            Lexicon.FUNC: (EnumBinaryOperation._member_names_, {"default": EnumBinaryOperation.ADD.name, "tooltip":"Arithmetic operation to perform"}),
            Lexicon.TYPE: (names_convert, {"default": names_convert[2],
                                           "tooltip":"Output type desired from resultant operation"}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
            Lexicon.X: ("FLOAT", {"default": 0, "tooltip":"Single value input"}),
            Lexicon.IN_A+"2": ("VEC2", {"default": (0,0),
                                      "label": [Lexicon.X, Lexicon.Y],
                                      "tooltip":"2-value vector"}),
            Lexicon.IN_A+"3": ("VEC3", {"default": (0,0,0),
                                      "label": [Lexicon.X, Lexicon.Y, Lexicon.Z],
                                      "tooltip":"3-value vector"}),
            Lexicon.IN_A+"4": ("VEC4", {"default": (0,0,0,0),
                                      "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
                                      "tooltip":"4-value vector"}),
            Lexicon.Y: ("FLOAT", {"default": 0, "tooltip":"Single value input"}),
            Lexicon.IN_B+"2": ("VEC2", {"default": (0,0),
                                      "label": [Lexicon.X, Lexicon.Y],
                                      "tooltip":"2-value vector"}),
            Lexicon.IN_B+"3": ("VEC3", {"default": (0,0,0),
                                      "label": [Lexicon.X, Lexicon.Y, Lexicon.Z],
                                      "tooltip":"3-value vector"}),
            Lexicon.IN_B+"4": ("VEC4", {"default": (0,0,0,0),
                                      "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
                                      "tooltip":"4-value vector"}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#-calc-op-binary")

    def run(self, **kw) -> tuple[bool]:
        results = []
        A = kw.get(Lexicon.IN_A, [None])
        B = kw.get(Lexicon.IN_B, [None])

        a_x = kw.get(Lexicon.X, [0])
        a_xy = parse_tuple(Lexicon.IN_A+"2", kw, EnumTupleType.FLOAT, (0, 0))
        a_xyz = parse_tuple(Lexicon.IN_A+"3", kw, EnumTupleType.FLOAT, (0, 0, 0))
        a_xyzw = parse_tuple(Lexicon.IN_A+"4", kw, EnumTupleType.FLOAT, (0, 0, 0, 0))

        b_x = kw.get(Lexicon.Y, [0])
        b_xy = parse_tuple(Lexicon.IN_B+"2", kw, EnumTupleType.FLOAT, (0, 0))
        b_xyz = parse_tuple(Lexicon.IN_B+"3", kw, EnumTupleType.FLOAT, (0, 0, 0))
        b_xyzw = parse_tuple(Lexicon.IN_B+"4", kw, EnumTupleType.FLOAT, (0, 0, 0, 0))

        op = kw.get(Lexicon.FUNC, [EnumBinaryOperation.ADD])
        typ = kw.get(Lexicon.TYPE, [EnumConvertType.FLOAT])
        flip = kw.get(Lexicon.FLIP, [False])
        params = [tuple(x) for x in zip_longest_fill(A, B, a_x, a_xy, a_xyz, a_xyzw,
                                                     b_x, b_xy, b_xyz, b_xyzw, op, typ, flip)]
        pbar = ProgressBar(len(params))
        for idx, (A, B, a_x, a_xy, a_xyz, a_xyzw,
                  b_x, b_xy, b_xyz, b_xyzw, op, typ, flip) in enumerate(params):

            typ = EnumConvertType[typ]
            if typ == EnumConvertType.BOOLEAN:
                typ_a, val_a, size_a = parse_type_value(typ, A, A if A is not None else a_x)
                typ_b, val_b, size_b = parse_type_value(typ, B, B if B is not None else b_x)
            elif typ in [EnumConvertType.INT, EnumConvertType.FLOAT]:
                typ_a, val_a, size_a = parse_type_value(typ, A, A if A is not None else a_x)
                typ_b, val_b, size_b = parse_type_value(typ, B, B if B is not None else b_x)
            elif typ in [EnumConvertType.VEC2, EnumConvertType.VEC2INT]:
                typ_a, val_a, size_a = parse_type_value(typ, A, A if A is not None else a_xy)
                typ_b, val_b, size_b = parse_type_value(typ, B, B if B is not None else b_xy)
            elif typ in [EnumConvertType.VEC3, EnumConvertType.VEC3INT]:
                typ_a, val_a, size_a = parse_type_value(typ, A, A if A is not None else a_xyz)
                typ_b, val_b, size_b = parse_type_value(typ, B, B if B is not None else b_xyz)
            else:
                typ_a, val_a, size_a = parse_type_value(typ, A, A if A is not None else a_xyzw)
                typ_b, val_b, size_b = parse_type_value(typ, B, B if B is not None else b_xyzw)
            val_a = convert_value(typ_a, val_a, size_a)
            if not isinstance(val_a, (list, set, tuple,)):
                val_a = [val_a]
            val_b = convert_value(typ_b, val_b, size_b)
            if not isinstance(val_b, (list, set, tuple,)):
                val_b = [val_b]
            if (short := len(val_a) - len(val_b)) > 0:
                val_b.extend([0] * short)
            convert = int if typ in [
                EnumConvertType.BOOLEAN,
                EnumConvertType.INT,
                EnumConvertType.VEC2INT,
                EnumConvertType.VEC3INT,
                EnumConvertType.VEC4INT, ] else float
            val_a = [convert(v) for v in val_a][:size_a]
            val_b = [convert(v) for v in val_b][:size_b]
            if flip:
                val_a, val_b = val_b, val_a

            match EnumBinaryOperation[op]:
                # VECTOR
                case EnumBinaryOperation.DOT_PRODUCT:
                    val = [sum(a * b for a, b in zip(val_a, val_b))]
                case EnumBinaryOperation.CROSS_PRODUCT:
                    val = [0, 0, 0]
                    if len(val_a) < 3 or len(val_b) < 3:
                        logger.warning("Cross product only defined for 3D vectors")
                    else:
                        val = [
                            val_a[1] * val_b[2] - val_a[2] * val_b[1],
                            val_a[2] * val_b[0] - val_a[0] * val_b[2],
                            val_a[0] * val_b[1] - val_a[1] * val_b[0]
                        ]

                # ARITHMETIC
                case EnumBinaryOperation.ADD:
                    val = [sum(pair) for pair in zip(val_a, val_b)]
                case EnumBinaryOperation.SUBTRACT:
                    val = [a - b for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.MULTIPLY:
                    val = [a * b for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.DIVIDE:
                    val = [a / b if b != 0 else 0 for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.DIVIDE_FLOOR:
                    val = [a // b if b != 0 else 0 for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.MODULUS:
                    val = [a % b if b != 0 else 0 for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.POWER:
                    val = [a ** b for a, b in zip(val_a, val_b)]

                case EnumBinaryOperation.MAXIMUM:
                    val = [max(val_a, val_b)]
                case EnumBinaryOperation.MINIMUM:
                    val = [min(val_a, val_b)]

                # BITS
                # case EnumBinaryOperation.BIT_NOT:
                case EnumBinaryOperation.BIT_AND:
                    val = [int(a) & int(b) for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.BIT_NAND:
                    val = [not(int(a) & int(b)) for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.BIT_OR:
                    val = [int(a) | int(b) for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.BIT_NOR:
                    val = [not(int(a) | int(b)) for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.BIT_XOR:
                    val = [int(a) ^ int(b) for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.BIT_XNOR:
                    val = [not(int(a) ^ int(b)) for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.BIT_LSHIFT:
                    val = [int(a) << int(b) for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.BIT_RSHIFT:
                    val = [int(a) >> int(b) for a, b in zip(val_a, val_b)]

                # GROUP
                case EnumBinaryOperation.UNION:
                    val = list(set(val_a) | set(val_b))
                case EnumBinaryOperation.INTERSECTION:
                    val = list(set(val_a) & set(val_b))
                case EnumBinaryOperation.DIFFERENCE:
                    val = list(set(val_a) - set(val_b))

            if len(val) == 0:
                val = [None]
            results.append(val)
            pbar.update_absolute(idx)
        return list(zip(*results))

class ValueNode(JOVBaseNode):
    NAME = "VALUE (JOV) 🧬"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Create a value for most types; also universal constants."
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.ANY, )
    OUTPUT_IS_LIST = (True, )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None, "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections."}),
            Lexicon.TYPE: (EnumConvertType._member_names_, {"default": EnumConvertType.BOOLEAN.name}),
            Lexicon.X: ("FLOAT", {"default": 0}),
            Lexicon.Y: ("FLOAT", {"default": 0}),
            Lexicon.Z: ("FLOAT", {"default": 0}),
            Lexicon.W: ("FLOAT", {"default": 0}),
            Lexicon.STRING: ("STRING", {"default": "", "dynamicPrompts": False, "multiline": True}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#%EF%B8%8F⃣-value")

    def run(self, **kw) -> tuple[bool]:
        raw = kw.get(Lexicon.IN_A, [None])
        typ = kw.get(Lexicon.TYPE, [EnumConvertType.BOOLEAN])
        x = kw.get(Lexicon.X, [None])
        y = kw.get(Lexicon.Y, [0])
        z = kw.get(Lexicon.Z, [0])
        w = kw.get(Lexicon.W, [0])
        params = [tuple(x) for x in zip_longest_fill(raw, typ, x, y, z, w)]
        results = []
        pbar = ProgressBar(len(params))
        for idx, (raw, typ, x, y, z, w) in enumerate(params):
            typ = EnumConvertType[typ]
            typ, val, size = parse_type_value(typ, raw, [x, y, z, w])
            val = convert_value(typ, val, size)
            results.append(val)
            pbar.update_absolute(idx)
        return (results, )

class LerpNode(JOVBaseNode):
    NAME = "LERP (JOV) 🔰"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Interpolate between two values with or without a smoothing."
    OUTPUT_IS_LIST = (True, )
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.ANY )
    SORT = 45

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {}),
            Lexicon.IN_B: (WILDCARD, {}),
            Lexicon.FLOAT: ("FLOAT", {"default": 0., "min": 0., "max": 1.0, "step": 0.001, "precision": 4, "round": 0.00001, "tooltip": "Blend Amount. 0 = full A, 1 = full B"}),
            Lexicon.EASE: (["NONE"] + EnumEase._member_names_, {"default": "NONE"}),
            Lexicon.TYPE: (EnumNumberType._member_names_, {"default": EnumNumberType.FLOAT.name, "tooltip": "Output As"})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#-lerp")

    def run(self, **kw) -> tuple[Any, Any]:
        a = kw.get(Lexicon.IN_A, [0])
        b = kw.get(Lexicon.IN_B, [0])
        pos = kw.get(Lexicon.FLOAT, [0.])
        op = kw.get(Lexicon.EASE, ["NONE"])
        typ = kw.get(Lexicon.TYPE, ["NONE"])
        value = []
        params = [tuple(x) for x in zip_longest_fill(a, b, pos, op, typ)]
        pbar = ProgressBar(len(params))
        for idx, (a, b, pos, op, typ) in enumerate(params):
            # make sure we only interpolate between the smallest "stride" we can
            size = min(len(a), len(b))
            typ = EnumNumberType[typ]
            ease = EnumEase[op]

            def same():
                val = 0.
                if op == "NONE":
                    val = b * pos + a * (1 - pos)
                else:
                    val = ease_op(ease, a, b, alpha=pos)
                return val

            if size == 3:
                same()
            elif size == 2:
                same()
            elif size == 1:
                same()

            pbar.update_absolute(idx)
        return (value, )
