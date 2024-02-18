"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Calculation
"""

import math
from enum import Enum
from typing import Any
from collections import Counter

from scipy.special import gamma
from loguru import logger

import comfy
# from server import PromptServer
import nodes

from Jovimetrix import JOV_HELP_URL, JOVBaseNode, IT_REQUIRED, WILDCARD, IT_FLIP
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import zip_longest_fill, convert_parameter, deep_merge_dict
from Jovimetrix.sup.anim import Ease, EnumEase
#from Jovimetrix.sup.image import cv2mask, cv2tensor, image_load, tensor2pil, \
#    pil2tensor, image_formats
# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/CALC"

# =============================================================================

class EnumNumberType(Enum):
    INT = 0
    FLOAT = 10

class EnumConvertType(Enum):
    STRING = 0
    BOOLEAN = 10
    INT = 20
    FLOAT   = 30
    VEC2 = 40
    VEC3 = 50
    VEC4 = 60
    # TUPLE = 70

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
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Perform a Unary Operation on an input."
    INPUT_IS_LIST = True
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN, )
    OUTPUT_IS_LIST = (True, )
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.FUNC: (EnumUnaryOperation._member_names_, {"default": EnumUnaryOperation.ABS.name})
        }}
        d = deep_merge_dict(IT_REQUIRED, d)
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#-calc-op-unary")

    def run(self, **kw) -> tuple[bool]:
        result = []
        data = kw.get(Lexicon.IN_A, [0])
        op = kw.get(Lexicon.FUNC, [EnumUnaryOperation.ABS])
        params = [tuple(x) for x in zip_longest_fill(data, op)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (data, op) in enumerate(params):
            typ, val = convert_parameter(data)
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
                    if len(val) == 1:
                        val = [math.sqrt(val[0] ** 2)]
                    else:
                        val = [math.sqrt(sum(x ** 2 for x in val))]
                case EnumUnaryOperation.NORMALIZE:
                    if len(val) == 1:
                        val = [1]
                    else:
                        m = math.sqrt(sum(x ** 2 for x in val))
                        val = [v / m for v in val]
                case EnumUnaryOperation.MAXIMUM:
                    if len(val) == 1:
                        val = [val]
                    else:
                        val = [max(val)]

                case EnumUnaryOperation.MINIMUM:
                    if len(val) == 1:
                        val = [val]
                    else:
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

            # Reconvert the values back to their original types
            val = [typ[i](v) for i, v in enumerate(val)]
            if len(val) == 1:
                result.append(val[0])
            else:
                result.append(tuple(val))
            # logger.debug("{} {}", result, val)
            pbar.update_absolute(idx)

        return (result, )

class CalcBinaryOPNode(JOVBaseNode):
    NAME = "CALC OP BINARY (JOV) 🌟"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Perform a Binary Operation on two inputs."
    INPUT_IS_LIST = True
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN, )
    OUTPUT_IS_LIST = (True, )
    SORT = 20

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.FUNC: (EnumBinaryOperation._member_names_, {"default": EnumBinaryOperation.ADD.name}),
            Lexicon.IN_B: (WILDCARD, {"default": None})
        }}
        d = deep_merge_dict(IT_REQUIRED, d, IT_FLIP)
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#-calc-op-binary")

    def run(self, **kw) -> tuple[bool]:
        result = []
        A = kw[Lexicon.IN_A]
        B = kw[Lexicon.IN_B]
        flip = kw[Lexicon.FLIP]
        op = kw[Lexicon.FUNC]
        params = [tuple(x) for x in zip_longest_fill(A, B, op, flip)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (a, b, op, flip) in enumerate(params):
            if type(a) == tuple and type(b) == tuple:
                if (short := len(a) - len(b)) > 0:
                    b = [i for i in b] + [0] * short

            typ_a, val_a = convert_parameter(a)
            _, val_b = convert_parameter(b)
            if flip:
                a, b = b, a

            match EnumBinaryOperation[op]:
                # VECTOR
                case EnumBinaryOperation.DOT_PRODUCT:
                    val = [sum(a * b for a, b in zip(a, b))]
                case EnumBinaryOperation.CROSS_PRODUCT:
                    if len(a) != 3 or len(b) != 3:
                        logger.warning("Cross product only defined for 3D vectors")
                        return [0, 0, 0]
                    return [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]
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

            val = [typ_a[i](v) for i, v in enumerate(val)]
            if len(val) == 1:
                result.append(val[0])
            else:
                result.append(tuple(val))
            # logger.debug("{} {}", result, val)
            pbar.update_absolute(idx)

        return (result, )

class ValueNode(JOVBaseNode):
    NAME = "VALUE (JOV) #️⃣"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Create a value for most types; also universal constants."
    INPUT_IS_LIST = True
    RETURN_TYPES = (WILDCARD, )
    OUTPUT_IS_LIST = (True, )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.TYPE: (EnumConvertType._member_names_, {"default": EnumConvertType.BOOLEAN.name}),
            Lexicon.X: ("FLOAT", {"default": 0}),
            Lexicon.Y: ("FLOAT", {"default": 0}),
            Lexicon.Z: ("FLOAT", {"default": 0}),
            Lexicon.W: ("FLOAT", {"default": 0})
        }}
        d = deep_merge_dict(IT_REQUIRED, d)
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#%EF%B8%8F⃣-value")

    def run(self, **kw) -> tuple[bool]:
        typ = kw.get(Lexicon.TYPE, [EnumConvertType.BOOLEAN])
        x = kw.get(Lexicon.X, [None])
        y = kw.get(Lexicon.Y, [0])
        z = kw.get(Lexicon.Z, [0])
        w = kw.get(Lexicon.W, [0])
        params = [tuple(x) for x in zip_longest_fill(typ, x, y, z, w)]
        logger.debug(params)
        results = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (typ, x, y, z, w) in enumerate(params):
            typ = EnumConvertType[typ]
            if typ == EnumConvertType.STRING:
                results.append("" if x is None else str(x))
                continue

            x = 0 if x is None else x
            match typ:
                case EnumConvertType.VEC2:
                    results.append((x, y,))
                case EnumConvertType.VEC3:
                    results.append((x, y, z,))
                case EnumConvertType.VEC4:
                    results.append((x, y, z, w,))
                case _:
                    results.append(x)

            pbar.update_absolute(idx)
        logger.debug(results)
        return (results, )

class ConvertNode(JOVBaseNode):
    NAME = "CONVERT (JOV) 🧬"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Convert INT, FLOAT, VEC*, STRING and BOOL."
    RETURN_TYPES = (WILDCARD,)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    SORT = 5

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.TYPE: (["STRING", "BOOLEAN", "INT", "FLOAT", "VEC2", "VEC3", "VEC4"], {"default": "BOOLEAN"})
        }}
        d = deep_merge_dict(IT_REQUIRED, d)
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#-convert")

    @staticmethod
    def convert(typ, val) -> tuple | tuple[Any]:
        size = len(val) if type(val) == tuple else 0
        if typ in ["STRING"]:
            if size > 0:
                return " ".join(str(val))
            return str(val)
        elif typ in ["FLOAT"]:
            if size > 0:
                return float(val[0])
            return float(val)
        elif typ == "BOOLEAN":
            if size > 0:
                return bool(val[0])
            return bool(val)
        elif typ == "INT":
            if size > 0:
                return int(val[0])
            return int(val)
        elif typ == "VEC2":
            if size > 1:
                return (val[0], val[1], )
            elif size > 0:
                return (val[0], val[0], )
            return (val, val, )
        elif typ == "VEC3":
            if size > 2:
                return (val[0], val[1], val[2], )
            elif size > 1:
                return (val[0], val[1], val[1], )
            elif size > 0:
                return (val[0], val[0], val[0], )
            return (val, val, val, )
        elif typ == "VEC4":
            if size > 3:
                return (val[0], val[1], val[2], val[3], )
            elif size > 2:
                return (val[0], val[1], val[2], val[2], )
            elif size > 1:
                return (val[0], val[1], val[1], val[1], )
            elif size > 0:
                return (val[0], val[0], val[0], val[0], )
            return (val, val, val, val, )
        else:
            return "nan"

    def run(self, **kw) -> tuple[bool]:
        results = []
        typ = kw.pop(Lexicon.TYPE, ["STRING"])
        values = kw.values()
        params = [tuple(x) for x in zip_longest_fill(typ, values)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (typ, values) in enumerate(params):
            result = []

            v = ''
            try: v = next(iter(values))
            except: pass

            if not isinstance(v, (list, set, tuple)):
                v = [v]

            for idx, val in enumerate(v):
                val_new = ConvertNode.convert(typ, val)
                result.append(val_new)

            results.append(result)
            pbar.update_absolute(idx)

        return results

class LerpNode(JOVBaseNode):
    NAME = "LERP (JOV) 🔰"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Interpolate between two values with or without a smoothing."
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.ANY )
    SORT = 45

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {}),
            Lexicon.IN_B: (WILDCARD, {}),
            Lexicon.FLOAT: ("FLOAT", {"default": 0., "min": 0., "max": 1.0, "step": 0.001, "precision": 4, "round": 0.00001, "tooltip": "Blend Amount. 0 = full A, 1 = full B"}),
            Lexicon.EASE: (["NONE"] + EnumEase._member_names_, {"default": "NONE"}),
            Lexicon.TYPE: (EnumNumberType._member_names_, {"default": EnumNumberType.FLOAT.name, "tooltip": "Output As"})
        }}
        d = deep_merge_dict(IT_REQUIRED, d)
        return Lexicon._parse(d, JOV_HELP_URL + "/CALC#-lerp")

    def run(self, **kw) -> tuple[Any, Any]:
        a = kw.get(Lexicon.IN_A, [0])
        b = kw.get(Lexicon.IN_B, [0])
        pos = kw.get(Lexicon.FLOAT, [0.])
        op = kw.get(Lexicon.EASE, ["NONE"])
        typ = kw.get(Lexicon.TYPE, ["NONE"])

        value = []
        params = [tuple(x) for x in zip_longest_fill(a, b, pos, op, typ)]
        pbar = comfy.utils.ProgressBar(len(params))
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
                    val = Ease.ease(ease, start=a, end=b, alpha=pos)
                return val

            if size == 3:
                same()
            elif size == 2:
                same()
            elif size == 1:
                same()

            pbar.update_absolute(idx)
        return (value, )
