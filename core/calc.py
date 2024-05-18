"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Calculation
"""

import sys
import math
from enum import Enum
from typing import Any, Tuple
from collections import Counter

import torch
import numpy as np
from scipy.special import gamma
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOVBaseNode, WILDCARD, JOV_WEB_RES_ROOT
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_param, parse_value, vector_swap, \
    zip_longest_fill, EnumConvertType, EnumSwizzle
from Jovimetrix.sup.anim import ease_op, EnumEase

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

class CalcUnaryOPNode(JOVBaseNode):
    NAME = "CALC OP UNARY (JOV) 🎲"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN,)
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {},
            "optional": {
                Lexicon.IN_A: (WILDCARD, {"default": None}),
                Lexicon.FUNC: (EnumUnaryOperation._member_names_, {"default": EnumUnaryOperation.ABS.name})
            }
        }
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> Tuple[bool]:
        results = []
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.VEC4, 0)
        op = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumUnaryOperation.ABS.name, enumType=EnumUnaryOperation)
        params = list(zip_longest_fill(A, op))
        pbar = ProgressBar(len(params))
        for idx, (A, op) in enumerate(params):
            typ = EnumConvertType.ANY
            if isinstance(A, (str, )):
                typ = EnumConvertType.STRING
            elif isinstance(A, (bool, )):
                typ = EnumConvertType.BOOLEAN
            elif isinstance(A, (int, )):
                typ = EnumConvertType.INT
            elif isinstance(A, (float, )):
                typ = EnumConvertType.FLOAT
            elif isinstance(A, (list, set, tuple,)):
                typ = EnumConvertType(len(A) * 10)
            elif isinstance(A, (dict,)):
                typ = EnumConvertType.DICT
            elif isinstance(A, (torch.Tensor,)):
                typ = EnumConvertType.IMAGE
            val = parse_value(A, typ, 0)
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
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN,)
    SORT = 20

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        names_convert = EnumConvertType._member_names_[:9]
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
            Lexicon.X: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize, "tooltip":"Single value input"}),
            Lexicon.IN_A+"2": ("VEC2", {"default": (0,0),
                                      "label": [Lexicon.X, Lexicon.Y],
                                      "tooltip":"2-value vector"}),
            Lexicon.IN_A+"3": ("VEC3", {"default": (0,0,0),
                                      "label": [Lexicon.X, Lexicon.Y, Lexicon.Z],
                                      "tooltip":"3-value vector"}),
            Lexicon.IN_A+"4": ("VEC4", {"default": (0,0,0,0),
                                      "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
                                      "tooltip":"4-value vector"}),
            Lexicon.Y: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize, "tooltip":"Single value input"}),
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

    def run(self, **kw) -> Tuple[bool]:
        results = []
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        B = parse_param(kw, Lexicon.IN_B, EnumConvertType.ANY, None)
        a_x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, 0)
        a_xy = parse_param(kw, Lexicon.IN_A+"2", EnumConvertType.VEC2, [(0, 0)])
        a_xyz = parse_param(kw, Lexicon.IN_A+"3", EnumConvertType.VEC3, [(0, 0, 0)])
        a_xyzw = parse_param(kw, Lexicon.IN_A+"4", EnumConvertType.VEC4, [(0, 0, 0, 0)])
        b_x = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, 0)
        b_xy = parse_param(kw, Lexicon.IN_B+"2", EnumConvertType.VEC2, [(0, 0)])
        b_xyz = parse_param(kw, Lexicon.IN_B+"3", EnumConvertType.VEC3, [(0, 0, 0)])
        b_xyzw = parse_param(kw, Lexicon.IN_B+"4", EnumConvertType.VEC4, [(0, 0, 0, 0)])
        op = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumBinaryOperation.ADD.name, enumType=EnumBinaryOperation)
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumConvertType.FLOAT.name, enumType=EnumConvertType)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(A, B, a_x, a_xy, a_xyz, a_xyzw,
                                  b_x, b_xy, b_xyz, b_xyzw, op, typ, flip))
        pbar = ProgressBar(len(params))
        for idx, (A, B, a_x, a_xy, a_xyz, a_xyzw,
                  b_x, b_xy, b_xyz, b_xyzw, op, typ, flip) in enumerate(params):

            # use everything as float for precision
            typ = EnumConvertType[typ]
            if typ in [EnumConvertType.VEC2, EnumConvertType.VEC2INT]:
                val_a = parse_value(A, EnumConvertType.VEC4, A if A is not None else a_xy)
                val_b = parse_value(B, EnumConvertType.VEC4, B if B is not None else b_xy)
            elif typ in [EnumConvertType.VEC3, EnumConvertType.VEC3INT]:
                val_a = parse_value(A, EnumConvertType.VEC4, A if A is not None else a_xyz)
                val_b = parse_value(B, EnumConvertType.VEC4, B if B is not None else b_xyz)
            elif typ in [EnumConvertType.VEC4, EnumConvertType.VEC4INT]:
                val_a = parse_value(A, EnumConvertType.VEC4, A if A is not None else a_xyzw)
                val_b = parse_value(B, EnumConvertType.VEC4, B if B is not None else b_xyzw)
            else:
                # logger.debug('val', A, B)
                val_a = parse_value(A, EnumConvertType.VEC4, A if A is not None else a_x)
                val_b = parse_value(B, EnumConvertType.VEC4, B if B is not None else a_x)

            if flip:
                val_a, val_b = val_b, val_a
            size = max(1, int(typ.value / 10))
            val_a = val_a[:size]
            val_b = val_b[:size]

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
                    val = max(val_a, val_b)
                case EnumBinaryOperation.MINIMUM:
                    val = min(val_a, val_b)

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

            # cast into correct type....
            val = parse_value(val, typ, val)
            if len(val) == 0:
                val = [0]
            results.append(val)
            pbar.update_absolute(idx)
        return (results,)

class ValueNode(JOVBaseNode):
    NAME = "VALUE (JOV) 🧬"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.ANY,)
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        typ = EnumConvertType._member_names_
        try: typ.pop(typ.index('IMAGE'))
        except: pass
        try: typ.pop(typ.index('LATENT'))
        except: pass
        d = {
            "required": {},
            "optional": {
                Lexicon.IN_A: (WILDCARD, {"default": None, "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
                Lexicon.TYPE: (EnumConvertType._member_names_, {"default": EnumConvertType.BOOLEAN.name}),
                Lexicon.X: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
                Lexicon.Y: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
                Lexicon.Z: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
                Lexicon.W: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
                Lexicon.STRING: ("STRING", {"default": "", "dynamicPrompts": False, "multiline": True}),
            }
        }
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> Tuple[bool]:
        x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, 0)
        y = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, 0)
        z = parse_param(kw, Lexicon.Z, EnumConvertType.FLOAT, 0)
        w = parse_param(kw, Lexicon.W, EnumConvertType.FLOAT, 0)
        raw = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumConvertType.BOOLEAN.name)
        params = list(zip_longest_fill(raw, typ, x, y, z, w))
        results = []
        pbar = ProgressBar(len(params))
        for idx, (raw, typ, x, y, z, w) in enumerate(params):
            typ = EnumConvertType[typ]
            val = parse_value(raw, typ, (x, y, z, w))
            results.append(val)
            pbar.update_absolute(idx)
        return (results,)

class LerpNode(JOVBaseNode):
    NAME = "LERP (JOV) 🔰"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.ANY,)
    SORT = 45

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {"tooltip": "Custom Start Point"}),
            Lexicon.IN_B: (WILDCARD, {"tooltip": "Custom End Point"}),
            Lexicon.FLOAT: ("FLOAT", {"default": 0., "min": 0., "max": 1.0, "step": 0.001, "precision": 4, "round": 0.00001, "tooltip": "Blend Amount. 0 = full A, 1 = full B"}),
            Lexicon.EASE: (["NONE"] + EnumEase._member_names_, {"default": "NONE"}),
            Lexicon.TYPE: (EnumNumberType._member_names_, {"default": EnumNumberType.FLOAT.name, "tooltip": "Output As"})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> Tuple[Any, Any]:
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.VEC4, [(0,0,0,0)], 0, 1)
        B = parse_param(kw, Lexicon.IN_B, EnumConvertType.VEC4, [(1,1,1,1)], 0, 1)
        alpha = parse_param(kw, Lexicon.FLOAT,EnumConvertType.FLOAT, 0, 0, 1)
        op = parse_param(kw, Lexicon.EASE, EnumConvertType.STRING, "NONE")
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumNumberType.FLOAT.name)
        values = []
        params = list(zip_longest_fill(A, B, alpha, op, typ))
        pbar = ProgressBar(len(params))
        for idx, (A, B, alpha, op, typ) in enumerate(params):
            # make sure we only interpolate between the longest "stride" we can
            size = min(3, max(len(A), len(B)))
            best_type = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]
            A = parse_param(A, best_type, A)
            B = parse_param(B, best_type, B)
            alpha = parse_param(alpha, best_type, [alpha])
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
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.ANY,)
    SORT = 65

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {}),
            Lexicon.IN_B: (WILDCARD, {}),
            Lexicon.SWAP_X: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_X.name}),
            Lexicon.X: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
            Lexicon.SWAP_Y: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_Y.name}),
            Lexicon.Y: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
            Lexicon.SWAP_Z: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_Z.name}),
            Lexicon.Z: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
            Lexicon.SWAP_W: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_W.name}),
            Lexicon.W: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw)  -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.IN_A, EnumConvertType.VEC4, [(0,0,0,0)], 0, 1)
        pB = parse_param(kw, Lexicon.IN_B, EnumConvertType.VEC4, [(0,0,0,0)], 0, 1)
        swap_x = parse_param(kw, Lexicon.SWAP_X, EnumConvertType.STRING, EnumSwizzle.A_X.name)
        x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, 0)
        swap_y = parse_param(kw, Lexicon.SWAP_Y, EnumConvertType.STRING, EnumSwizzle.A_Y.name)
        y = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, 0)
        swap_z = parse_param(kw, Lexicon.SWAP_Z, EnumConvertType.STRING, EnumSwizzle.A_W.name)
        z = parse_param(kw, Lexicon.Z, EnumConvertType.FLOAT, 0)
        swap_w = parse_param(kw, Lexicon.SWAP_W, EnumConvertType.STRING, EnumSwizzle.A_Z.name)
        w = parse_param(kw, Lexicon.W, EnumConvertType.FLOAT, 0)
        params = list(zip_longest_fill(pA, pB, swap_x, x, swap_y, y, swap_z, z, swap_w, w))
        results = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, swap_x, x, swap_y, y, swap_z, z, swap_w, w) in enumerate(params):
            swap_x = EnumSwizzle[swap_x]
            swap_y = EnumSwizzle[swap_y]
            swap_z = EnumSwizzle[swap_z]
            swap_w = EnumSwizzle[swap_w]
            val = vector_swap(pA, pB, swap_x, x, swap_y, y, swap_z, z, swap_w, w)
            results.append(val)
            pbar.update_absolute(idx)
        return (results,)
