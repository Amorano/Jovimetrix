"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Calculation
"""

import math
from enum import Enum
from typing import Any, List
from collections import Counter

import numpy as np
from scipy.special import gamma
from loguru import logger
import torch

from comfy.utils import ProgressBar

from Jovimetrix import JOVBaseNode, WILDCARD, load_help
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_parameter, parse_parameter, zip_longest_fill, EnumConvertType
from Jovimetrix.sup.anim import ease_op, EnumEase
from Jovimetrix.sup.image import  channel_solid, channel_swap, cv2tensor_full, \
    tensor2cv, EnumImageType, EnumSwizzle

# =============================================================================

JOV_CATEGORY = "CALC"

# =============================================================================

class EnumNumberType(Enum):
    INT = 0
    FLOAT = 10

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

"""
def parse_parameter(typ:EnumConvertType, A:Any, default:tuple=(0,0,0,0)) -> List[Any]:
    # Converts a number into a 1, 2, 3 or 4 vector array
    if not isinstance(default, (list, set, tuple,)):
        default = [default]
    size = max(1, int(typ.value / 10))
    if A is None:
        val = default
    elif isinstance(A, (torch.Tensor,)):
        val = list(A.size())[1:4] + [val[0]]
    else:
        val = A
    if isinstance(val, (set, tuple,)):
        val = list(val)
    if not isinstance(val, (list,)):
        val = [val]
    last = val[-1]
    pos = len(val)
    for x in range(size - pos):
        idx = pos + x
        val.append(default[idx] if idx < len(default) else last)
    return val[:size]

def convert_value(typ:EnumConvertType, val:Any) -> Any:
    if typ == EnumConvertType.STRING:
        return ", ".join([str(v) for v in val])
    elif typ == EnumConvertType.BOOLEAN:
        return bool(val[0])

    size = max(1, int(typ.value / 10))
    if typ in [EnumConvertType.FLOAT, EnumConvertType.VEC2,
                    EnumConvertType.VEC3, EnumConvertType.VEC4]:
        val = [round(float(v), 12) for v in val]
    else:
        val = [int(v) for v in val]
    val += [val[-1]] * (size - len(val))
    return val[:size]
"""

# =============================================================================

class CalcUnaryOPNode(JOVBaseNode):
    NAME = "CALC OP UNARY (JOV) 🎲"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-calc-op-unary"
    DESC = "Perform a Unary Operation on an input."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
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
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[bool]:
        results = []
        A = kw.get(Lexicon.IN_A, [0])
        op = kw.get(Lexicon.FUNC, [EnumUnaryOperation.ABS])
        params = [tuple(x) for x in zip_longest_fill(A, op)]
        pbar = ProgressBar(len(params))
        for idx, (A, op) in enumerate(params):
            val = parse_parameter(EnumConvertType.VEC4, A)
            val = [float(v) for v in val]
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
            convert = int if isinstance(A, (bool, int, np.uint8, np.uint16, np.uint32, np.uint64)) else float
            results.append([convert(v) for v in val])
            pbar.update_absolute(idx)
        return (results,)

class CalcBinaryOPNode(JOVBaseNode):
    NAME = "CALC OP BINARY (JOV) 🌟"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-calc-op-binary"
    DESC = "Perform a Binary Operation on two inputs."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
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
                                      "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
            Lexicon.IN_B: (WILDCARD, {"default": None,
                                      "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
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
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[bool]:
        results = []
        A = kw.get(Lexicon.IN_A, [None])
        B = kw.get(Lexicon.IN_B, [None])
        a_x = parse_parameter(Lexicon.X, kw, (0,), EnumConvertType.FLOAT)
        a_xy = parse_parameter(Lexicon.IN_A+"2", kw, (0, 0), EnumConvertType.FLOAT)
        a_xyz = parse_parameter(Lexicon.IN_A+"3", kw, (0, 0, 0), EnumConvertType.FLOAT)
        a_xyzw = parse_parameter(Lexicon.IN_A+"4", kw, (0, 0, 0, 0), EnumConvertType.FLOAT)
        b_x = parse_parameter(Lexicon.Y, kw, (0,), EnumConvertType.FLOAT)
        b_xy = parse_parameter(Lexicon.IN_B+"2", kw, (0, 0), EnumConvertType.FLOAT)
        b_xyz = parse_parameter(Lexicon.IN_B+"3", kw, (0, 0, 0), EnumConvertType.FLOAT)
        b_xyzw = parse_parameter(Lexicon.IN_B+"4", kw, (0, 0, 0, 0), EnumConvertType.FLOAT)
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
                val_a = parse_parameter(typ, A, A if A is not None else a_x)
                val_b = parse_parameter(typ, B, B if B is not None else b_x)
            elif typ in [EnumConvertType.INT, EnumConvertType.FLOAT]:
                val_a = parse_parameter(typ, A, A if A is not None else a_x)
                val_b = parse_parameter(typ, B, B if B is not None else b_x)
            elif typ in [EnumConvertType.VEC2, EnumConvertType.VEC2INT]:
                val_a = parse_parameter(typ, A, A if A is not None else a_xy)
                val_b = parse_parameter(typ, B, B if B is not None else b_xy)
            elif typ in [EnumConvertType.VEC3, EnumConvertType.VEC3INT]:
                val_a = parse_parameter(typ, A, A if A is not None else a_xyz)
                val_b = parse_parameter(typ, B, B if B is not None else b_xyz)
            else:
                val_a = parse_parameter(typ, A, A if A is not None else a_xyzw)
                val_b = parse_parameter(typ, B, B if B is not None else b_xyzw)

            val_a = [float(v) for v in val_a]
            val_b = [float(v) for v in val_b]
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
                val = [0]
            results.append(tuple(val))
            pbar.update_absolute(idx)
        return (results,)

class ValueNode(JOVBaseNode):
    NAME = "VALUE (JOV) 🧬"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-value"
    DESC = "Create a value for most types; also universal constants."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.ANY, )
    OUTPUT_IS_LIST = (True, )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None, "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
            Lexicon.TYPE: (EnumConvertType._member_names_, {"default": EnumConvertType.BOOLEAN.name}),
            Lexicon.X: ("FLOAT", {"default": 0}),
            Lexicon.Y: ("FLOAT", {"default": 0}),
            Lexicon.Z: ("FLOAT", {"default": 0}),
            Lexicon.W: ("FLOAT", {"default": 0}),
            Lexicon.STRING: ("STRING", {"default": "", "dynamicPrompts": False, "multiline": True}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[bool]:
        raw = kw.get(Lexicon.IN_A, [None])
        typ = kw.get(Lexicon.TYPE, [EnumConvertType.BOOLEAN])
        x = kw.get(Lexicon.X, [0])
        y = kw.get(Lexicon.Y, [0])
        z = kw.get(Lexicon.Z, [0])
        w = kw.get(Lexicon.W, [0])
        params = [tuple(x) for x in zip_longest_fill(raw, typ, x, y, z, w)]
        results = []
        pbar = ProgressBar(len(params))
        for idx, (raw, typ, x, y, z, w) in enumerate(params):
            typ = EnumConvertType[typ]
            val = parse_parameter(typ, raw, [x, y, z, w])
            # val = convert_value(typ, val)
            if isinstance(val, (set, tuple,)):
                val = list(val)
            if not isinstance(val, (list,)):
                val = [val]
            results.append(tuple(val))
            pbar.update_absolute(idx)
        return (results,)

class LerpNode(JOVBaseNode):
    NAME = "LERP (JOV) 🔰"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-lerp"
    DESC = "Interpolate between two values with or without a smoothing."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    OUTPUT_IS_LIST = (True, )
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.ANY )
    SORT = 45

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {"tooltip": "Custom Start Point"}),
            Lexicon.IN_B: (WILDCARD, {"tooltip": "Custom End Point"}),
            #Lexicon.FLOAT_A: ("FLOAT", {"default": 0., "min": 0., "max": 1.0, "step": 0.001, "precision": 4, "round": 0.00001, "tooltip": "Custom Start Point"}),
            #Lexicon.FLOAT+"1": ("FLOAT", {"default": 0., "min": 0., "max": 1.0, "step": 0.001, "precision": 4, "round": 0.00001, "tooltip": "Custom End point"}),
            Lexicon.FLOAT: ("FLOAT", {"default": 0., "min": 0., "max": 1.0, "step": 0.001, "precision": 4, "round": 0.00001, "tooltip": "Blend Amount. 0 = full A, 1 = full B"}),
            Lexicon.EASE: (["NONE"] + EnumEase._member_names_, {"default": "NONE"}),
            Lexicon.TYPE: (EnumNumberType._member_names_, {"default": EnumNumberType.FLOAT.name, "tooltip": "Output As"})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[Any, Any]:
        A = parse_parameter(Lexicon.IN_A, kw, [0], EnumConvertType.FLOAT)
        B = parse_parameter(Lexicon.IN_B, kw, [0], EnumConvertType.FLOAT)
        alpha = parse_parameter(Lexicon.FLOAT, kw, [0], EnumConvertType.FLOAT)
        op = kw.get(Lexicon.EASE, ["NONE"])
        typ = kw.get(Lexicon.TYPE, ["NONE"])
        values = []
        params = [tuple(x) for x in zip_longest_fill(A, B, alpha, op, typ)]
        pbar = ProgressBar(len(params))
        for idx, (A, B, alpha, op, typ) in enumerate(params):
            # make sure we only interpolate between the longest "stride" we can
            size = max(3, max(len(A), len(B)))
            best_type = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]
            A = parse_parameter(best_type, A)
            B = parse_parameter(best_type, B)
            alpha = parse_parameter(best_type, alpha)
            if op == "NONE":
                val = [B[x] * alpha[x] + A[x] * (1 - alpha[x]) for x in range(size)]
            else:
                ease = EnumEase[op]
                val = [ease_op(ease, A[x], B[x], alpha=alpha[x]) for x in range(size)]

            typ = EnumNumberType[typ]
            if typ == EnumNumberType.FLOAT:
                val = [float(v) for v in val]
            else:
                val = [int(v) for v in val]
            values.append(val)
            pbar.update_absolute(idx)
        return (values, )

class SwapNode(JOVBaseNode):
    NAME = "SWAP (JOV) 😵"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-swap"
    DESC = "Swap vector positions within a vector or with another vector input."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    SORT = 55

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.A: (WILDCARD, {}),
            Lexicon.B: (WILDCARD, {}),
            Lexicon.SWAP_X: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_X.name}),
            Lexicon.X: ("FLOAT", {"default": 0}),
            Lexicon.SWAP_Y: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_Y.name}),
            Lexicon.Y: ("FLOAT", {"default": 0}),
            Lexicon.SWAP_Z: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_Z.name}),
            Lexicon.Z: ("FLOAT", {"default": 0}),
            Lexicon.SWAP_W: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_W.name}),
            Lexicon.W: ("FLOAT", {"default": 0})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        pA = parse_parameter(Lexicon.A, kw, (0, 0, 0, 0))
        pB = parse_parameter(kw.get(Lexicon.B, None))
        swap_x = kw.get(Lexicon.SWAP_X, [EnumSwizzle.PASSTHRU])
        x = kw.get(Lexicon.X, [0])
        swap_y = kw.get(Lexicon.SWAP_Y, [EnumSwizzle.PASSTHRU])
        y = kw.get(Lexicon.Y, [0])
        swap_z = kw.get(Lexicon.SWAP_Z, [EnumSwizzle.PASSTHRU])
        z = kw.get(Lexicon.Z, [0])
        swap_w = kw.get(Lexicon.SWAP_W, [EnumSwizzle.PASSTHRU])
        w = kw.get(Lexicon.W, [0])
        params = [tuple(x) for x in zip_longest_fill(pA, pB, x, swap_x, y, swap_y,
                                                     z, swap_z, w, swap_w)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, r, swap_r, g, swap_g, b, swap_b, a, swap_a) in enumerate(params):
            pA = tensor2cv(pA)
            h, w = pA.shape[:2]
            pB = tensor2cv(pB, width=w, height=h)
            out = channel_solid(w, h, (r,g,b,a), EnumImageType.BGRA)

            def swapper(swap_out:EnumSwizzle, swap_in:EnumSwizzle) -> np.ndarray[Any]:
                target = out
                swap_in = EnumSwizzle[swap_in]
                if swap_in in [EnumSwizzle.RED_A, EnumSwizzle.GREEN_A,
                            EnumSwizzle.BLUE_A, EnumSwizzle.ALPHA_A]:
                    target = pA
                elif swap_in != EnumSwizzle.CONSTANT:
                    target = pB
                if swap_in != EnumSwizzle.CONSTANT:
                    target = channel_swap(pA, swap_out, target, swap_in)
                return target

            out[:,:,0] = swapper(EnumSwizzle.BLUE_A, swap_b)[:,:,0]
            out[:,:,1] = swapper(EnumSwizzle.GREEN_A, swap_g)[:,:,1]
            out[:,:,2] = swapper(EnumSwizzle.RED_A, swap_r)[:,:,2]
            out[:,:,3] = swapper(EnumSwizzle.ALPHA_A, swap_a)[:,:,3]
            images.append(cv2tensor_full(out))
            pbar.update_absolute(idx)
        return list(zip(*images))
