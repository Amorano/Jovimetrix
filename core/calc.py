""" Jovimetrix - Calculation """

import sys
import math
import struct
from enum import Enum
from typing import Any
from collections import Counter

import torch
from scipy.special import gamma

from comfy.utils import ProgressBar

from cozy_comfyui import \
    logger, \
    TensorType, InputType, EnumConvertType, \
    deep_merge, parse_dynamic, parse_param, parse_value, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_ANY, COZY_TYPE_NUMERICAL, COZY_TYPE_FULL, \
    CozyBaseNode

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

JOV_CATEGORY = "CALC"

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

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
    # WEIRD ONES
    BASE = 90

class EnumComparison(Enum):
    EQUAL = 0
    NOT_EQUAL = 1
    LESS_THAN = 2
    LESS_THAN_EQUAL = 3
    GREATER_THAN = 4
    GREATER_THAN_EQUAL = 5
    # LOGIC
    # NOT = 10
    AND = 20
    NAND = 21
    OR = 22
    NOR = 23
    XOR = 24
    XNOR = 25
    # TYPE
    IS = 80
    IS_NOT = 81
    # GROUPS
    IN = 82
    NOT_IN = 83

class EnumConvertString(Enum):
    SPLIT = 10
    JOIN = 30
    FIND = 40
    REPLACE = 50
    SLICE = 70  # start - end - step  = -1, -1, 1

class EnumSwizzle(Enum):
    A_X = 0
    A_Y = 10
    A_Z = 20
    A_W = 30
    B_X = 9
    B_Y = 11
    B_Z = 21
    B_W = 31
    CONSTANT = 40

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

# Dictionary to map each operation to its corresponding function
OP_UNARY = {
    EnumUnaryOperation.ABS: lambda x: math.fabs(x),
    EnumUnaryOperation.FLOOR: lambda x: math.floor(x),
    EnumUnaryOperation.CEIL: lambda x: math.ceil(x),
    EnumUnaryOperation.SQRT: lambda x: math.sqrt(x),
    EnumUnaryOperation.SQUARE: lambda x: math.pow(x, 2),
    EnumUnaryOperation.LOG: lambda x: math.log(x) if x != 0 else -math.inf,
    EnumUnaryOperation.LOG10: lambda x: math.log10(x) if x != 0 else -math.inf,
    EnumUnaryOperation.SIN: lambda x: math.sin(x),
    EnumUnaryOperation.COS: lambda x: math.cos(x),
    EnumUnaryOperation.TAN: lambda x: math.tan(x),
    EnumUnaryOperation.NEGATE: lambda x: -x,
    EnumUnaryOperation.RECIPROCAL: lambda x: 1 / x if x != 0 else 0,
    EnumUnaryOperation.FACTORIAL: lambda x: math.factorial(abs(int(x))),
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
    EnumUnaryOperation.GAMMA: lambda x: gamma(x) if x > 0 else 0,
}

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def to_bits(value: Any):
    if isinstance(value, int):
        return bin(value)[2:]
    elif isinstance(value, float):
        packed = struct.pack('>d', value)
        return ''.join(f'{byte:08b}' for byte in packed)
    elif isinstance(value, str):
        return ''.join(f'{ord(c):08b}' for c in value)
    else:
        raise TypeError(f"Unsupported type: {type(value)}")

def vector_swap(pA: Any, pB: Any, swap_x: EnumSwizzle, swap_y:EnumSwizzle,
                swap_z:EnumSwizzle, swap_w:EnumSwizzle, default:list[float]) -> list[float]:
    """Swap out a vector's values with another vector's values, or a constant fill."""

    def parse(target, targetB, swap, val) -> float:
        if swap == EnumSwizzle.CONSTANT:
            return val
        if swap in [EnumSwizzle.B_X, EnumSwizzle.B_Y, EnumSwizzle.B_Z, EnumSwizzle.B_W]:
            target = targetB
        swap = int(swap.value / 10)
        return target[swap]

    while len(pA) < 4:
        pA.append(0)

    while len(pB) < 4:
        pB.append(0)

    while len(default) < 4:
        default.append(0)

    return [
        parse(pA, pB, swap_x, default[0]),
        parse(pA, pB, swap_y, default[1]),
        parse(pA, pB, swap_z, default[2]),
        parse(pA, pB, swap_w, default[3])
    ]

# ==============================================================================
# === CLASS ===
# ==============================================================================

class BitSplitNode(CozyBaseNode):
    NAME = "BIT SPLIT (JOV) ⭄"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, "BOOLEAN",)
    RETURN_NAMES = ("BIT", "BOOL",)
    OUTPUT_IS_LIST = (True, True,)
    OUTPUT_TOOLTIPS = (
        "Bits as Numerical output (0 or 1)",
        "Bits as Boolean output (True or False)"
    )
    DESCRIPTION = """
Split an input into separate bits.
BOOL, INT and FLOAT use their numbers,
STRING is treated as a list of CHARACTER.
IMAGE and MASK will return a TRUE bit for any non-black pixel, as a stream of bits for all pixels in the image.
"""
    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.VALUE: (COZY_TYPE_NUMERICAL, {
                    "default": None,
                    "tooltip": "Value to convert into bits"}),
                Lexicon.BITS: ("INT", {
                    "default": 8, "min": 0, "max": 64,
                    "tooltip": "Number of output bits requested"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[list[int], list[bool]]:
        value = parse_param(kw, Lexicon.VALUE, EnumConvertType.LIST, 0)
        bits = parse_param(kw, Lexicon.BITS, EnumConvertType.INT, 8)
        params = list(zip_longest_fill(value, bits))
        pbar = ProgressBar(len(params))
        results = []
        for idx, (value, bits) in enumerate(params):
            bit_repr = to_bits(value[0])[::-1]
            if bits > 0:
                if len(bit_repr) > bits:
                    bit_repr = bit_repr[0:bits]
                else:
                    bit_repr = bit_repr.ljust(bits, '0')

            int_bits = []
            bool_bits = []
            for b in bit_repr:
                bit = int(b)
                int_bits.append(bit)
                bool_bits.append(bool(bit))
            results.append([int_bits, bool_bits])
            pbar.update_absolute(idx)
        return *list(zip(*results)),

class ComparisonNode(CozyBaseNode):
    NAME = "COMPARISON (JOV) 🕵🏽"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, COZY_TYPE_ANY,)
    RETURN_NAMES = ("OUT", "VAL",)
    OUTPUT_IS_LIST = (True, True,)
    OUTPUT_TOOLTIPS = (
        "Outputs the input at PASS or FAIL depending the evaluation",
        "The comparison result value"
    )
    DESCRIPTION = """
Evaluates two inputs (A and B) with a specified comparison operators and optional values for successful and failed comparisons. The node performs the specified operation element-wise between corresponding elements of A and B. If the comparison is successful for all elements, it returns the success value; otherwise, it returns the failure value. The node supports various comparison operators such as EQUAL, GREATER_THAN, LESS_THAN, AND, OR, IS, IN, etc.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IN_A: (COZY_TYPE_NUMERICAL, {
                    "default": 0,
                    "tooltip":"First value to compare"}),
                Lexicon.IN_B: (COZY_TYPE_NUMERICAL, {
                    "default": 0,
                    "tooltip":"Second value to compare"}),
                Lexicon.SUCCESS: (COZY_TYPE_ANY, {
                    "default": 0,
                    "tooltip": "Sent to OUT on a successful condition"}),
                Lexicon.FAIL: (COZY_TYPE_ANY, {
                    "default": 0,
                    "tooltip": "Sent to OUT on a failure condition"}),
                Lexicon.FUNCTION: (EnumComparison._member_names_, {
                    "default": EnumComparison.EQUAL.name,
                    "tooltip": "Comparison function. Sends the data in PASS on successful comparison to OUT, otherwise sends the value in FAIL"}),
                Lexicon.SWAP: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the A and B inputs"}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the PASS and FAIL inputs"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[Any, Any]:
        in_a = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, 0)
        in_b = parse_param(kw, Lexicon.IN_B, EnumConvertType.ANY, 0)
        size = max(len(in_a), len(in_b))
        good = parse_param(kw, Lexicon.SUCCESS, EnumConvertType.ANY, 0)[:size]
        fail = parse_param(kw, Lexicon.FAIL, EnumConvertType.ANY, 0)[:size]
        op = parse_param(kw, Lexicon.FUNCTION, EnumComparison, EnumComparison.EQUAL.name)[:size]
        swap = parse_param(kw, Lexicon.SWAP, EnumConvertType.BOOLEAN, False)[:size]
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)[:size]
        params = list(zip_longest_fill(in_a, in_b, good, fail, op, swap, invert))
        pbar = ProgressBar(len(params))
        vals = []
        results = []
        for idx, (A, B, good, fail, op, swap, invert) in enumerate(params):
            if not isinstance(A, (tuple, list,)):
                A = [A]
            if not isinstance(B, (tuple, list,)):
                B = [B]

            size = min(4, max(len(A), len(B))) - 1
            typ = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]

            val_a = parse_value(A, typ, [A[-1]] * size)
            if not isinstance(val_a, (list,)):
                val_a = [val_a]

            val_b = parse_value(B, typ, [B[-1]] * size)
            if not isinstance(val_b, (list,)):
                val_b = [val_b]

            if swap:
                val_a, val_b = val_b, val_a

            match op:
                case EnumComparison.EQUAL:
                    val = [a == b for a, b in zip(val_a, val_b)]
                case EnumComparison.GREATER_THAN:
                    val = [a > b for a, b in zip(val_a, val_b)]
                case EnumComparison.GREATER_THAN_EQUAL:
                    val = [a >= b for a, b in zip(val_a, val_b)]
                case EnumComparison.LESS_THAN:
                    val = [a < b for a, b in zip(val_a, val_b)]
                case EnumComparison.LESS_THAN_EQUAL:
                    val = [a <= b for a, b in zip(val_a, val_b)]
                case EnumComparison.NOT_EQUAL:
                    val = [a != b for a, b in zip(val_a, val_b)]
                # LOGIC
                # case EnumBinaryOperation.NOT = 10
                case EnumComparison.AND:
                    val = [a and b for a, b in zip(val_a, val_b)]
                case EnumComparison.NAND:
                    val = [not(a and b) for a, b in zip(val_a, val_b)]
                case EnumComparison.OR:
                    val = [a or b for a, b in zip(val_a, val_b)]
                case EnumComparison.NOR:
                    val = [not(a or b) for a, b in zip(val_a, val_b)]
                case EnumComparison.XOR:
                    val = [(a and not b) or (not a and b) for a, b in zip(val_a, val_b)]
                case EnumComparison.XNOR:
                    val = [not((a and not b) or (not a and b)) for a, b in zip(val_a, val_b)]
                # IDENTITY
                case EnumComparison.IS:
                    val = [a is b for a, b in zip(val_a, val_b)]
                case EnumComparison.IS_NOT:
                    val = [a is not b for a, b in zip(val_a, val_b)]
                # GROUP
                case EnumComparison.IN:
                    val = [a in val_b for a in val_a]
                case EnumComparison.NOT_IN:
                    val = [a not in val_b for a in val_a]

            output = all([bool(v) for v in val])
            if invert:
                output = not output

            output = good if output == True else fail
            results.append([output, val])
            pbar.update_absolute(idx)

        outs, vals = zip(*results)
        if isinstance(outs[0], (TensorType,)):
            if len(outs) > 1:
                outs = torch.stack(outs)
            else:
                outs = outs[0].unsqueeze(0)
            outs = [outs]
        else:
            outs = list(outs)
        return outs, *vals,

class LerpNode(CozyBaseNode):
    NAME = "LERP (JOV) 🔰"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY,)
    RETURN_NAMES = ("❔",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        f"Output can vary depending on the type chosen in the {"TYPE"} parameter",
    )
    DESCRIPTION = """
Calculate linear interpolation between two values or vectors based on a blending factor (alpha).

The node accepts optional start (IN_A) and end (IN_B) points, a blending factor (FLOAT), and various input types for both start and end points, such as single values (X, Y), 2-value vectors (IN_A2, IN_B2), 3-value vectors (IN_A3, IN_B3), and 4-value vectors (IN_A4, IN_B4).

Additionally, you can specify the easing function (EASE) and the desired output type (TYPE). It supports various easing functions for smoother transitions.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IN_A: (COZY_TYPE_NUMERICAL, {
                    "tooltip": "Custom Start Point"}),
                Lexicon.IN_B: (COZY_TYPE_NUMERICAL, {
                    "tooltip": "Custom End Point"}),
                Lexicon.ALPHA: ("VEC4", {
                    "default": (0.5, 0.5, 0.5, 0.5), "mij": 0, "maj": 1,}),
                Lexicon.TYPE: (EnumConvertType._member_names_[:6], {
                    "default": EnumConvertType.FLOAT.name,
                    "tooltip": "Output type desired from resultant operation"}),
                Lexicon.DEFAULT_A: ("VEC4", {
                    "default": (0, 0, 0, 0)}),
                Lexicon.DEFAULT_B: ("VEC4", {
                    "default": (1,1,1,1)})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[Any, Any]:
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, 0)
        B = parse_param(kw, Lexicon.IN_B, EnumConvertType.ANY, 0)
        alpha = parse_param(kw, Lexicon.ALPHA,EnumConvertType.VEC4, (0.5,0.5,0.5,0.5))
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType, EnumConvertType.FLOAT.name)
        a_xyzw = parse_param(kw, Lexicon.DEFAULT_A, EnumConvertType.VEC4, (0, 0, 0, 0))
        b_xyzw = parse_param(kw, Lexicon.DEFAULT_B, EnumConvertType.VEC4, (1, 1, 1, 1))
        values = []
        params = list(zip_longest_fill(A, B, alpha, typ, a_xyzw, b_xyzw))
        pbar = ProgressBar(len(params))
        for idx, (A, B, alpha, typ, a_xyzw, b_xyzw) in enumerate(params):
            size = int(typ.value / 10)

            if A is None:
                A = a_xyzw[:size]
            if B is None:
                B = b_xyzw[:size]

            val_a = parse_value(A, EnumConvertType.VEC4, a_xyzw)
            val_b = parse_value(B, EnumConvertType.VEC4, b_xyzw)
            alpha = parse_value(alpha, EnumConvertType.VEC4, alpha)

            if size > 1:
                val_a = val_a[:size + 1]
                val_b = val_b[:size + 1]
            else:
                val_a = [val_a[0]]
                val_b = [val_b[0]]

            val = [val_b[x] * alpha[x] + val_a[x] * (1 - alpha[x]) for x in range(size)]
            convert = int if "INT" in typ.name else float
            ret = []
            for v in val:
                try:
                    ret.append(convert(v))
                except OverflowError:
                    ret.append(0)
                except Exception as e:
                    ret.append(0)
            val = ret[0] if size == 1 else ret[:size+1]
            values.append(val)
            pbar.update_absolute(idx)
        return [values]

class OPUnaryNode(CozyBaseNode):
    NAME = "OP UNARY (JOV) 🎲"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY,)
    RETURN_NAMES = ("❔",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Output type will match the input type",
    )
    DESCRIPTION = """
Perform single function operations like absolute value, mean, median, mode, magnitude, normalization, maximum, or minimum on input values.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        typ = EnumConvertType._member_names_[:6]
        d = deep_merge(d, {
            "optional": {
                Lexicon.IN_A: (COZY_TYPE_FULL, {
                    "default": 0}),
                Lexicon.FUNCTION: (EnumUnaryOperation._member_names_, {
                    "default": EnumUnaryOperation.ABS.name}),
                Lexicon.TYPE: (typ, {
                    "default": EnumConvertType.FLOAT.name,}),
                Lexicon.DEFAULT_A: ("VEC4", {
                    "default": (0,0,0,0), "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "precision": 2,
                    "label": ["X", "Y", "Z", "W"]})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[bool]:
        results = []
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, 0)
        op = parse_param(kw, Lexicon.FUNCTION, EnumUnaryOperation, EnumUnaryOperation.ABS.name)
        out = parse_param(kw, Lexicon.TYPE, EnumConvertType, EnumConvertType.FLOAT.name)
        a_xyzw = parse_param(kw, Lexicon.DEFAULT_A, EnumConvertType.VEC4, (0, 0, 0, 0))
        params = list(zip_longest_fill(A, op, out, a_xyzw))
        pbar = ProgressBar(len(params))
        for idx, (A, op, out, a_xyzw) in enumerate(params):
            if not isinstance(A, (list, tuple,)):
                A = [A]
            best_type = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][len(A)-1]
            val = parse_value(A, best_type, a_xyzw)
            val = parse_value(val, EnumConvertType.VEC4, a_xyzw)
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
                        if m > 0:
                            val = [v / m for v in val]
                        else:
                            val = [0] * len(val)
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
                            logger.error(f"{e} :: {op}")
                            v = 0
                        ret.append(v)
                    val = ret

            val = parse_value(val, out, 0)
            results.append(val)
            pbar.update_absolute(idx)
        return (results,)

class OPBinaryNode(CozyBaseNode):
    NAME = "OP BINARY (JOV) 🌟"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY,)
    RETURN_NAMES = ("❔",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Output type will match the input type",
    )
    DESCRIPTION = """
Execute binary operations like addition, subtraction, multiplication, division, and bitwise operations on input values, supporting various data types and vector sizes.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        names_convert = EnumConvertType._member_names_[:6]
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IN_A: (COZY_TYPE_FULL, {
                    "default": None}),
                Lexicon.IN_B: (COZY_TYPE_FULL, {
                    "default": None}),
                Lexicon.FUNCTION: (EnumBinaryOperation._member_names_, {
                    "default": EnumBinaryOperation.ADD.name,}),
                Lexicon.TYPE: (names_convert, {
                    "default": names_convert[2],
                    "tooltip":"Output type desired from resultant operation"}),
                Lexicon.SWAP: ("BOOLEAN", {
                    "default": False}),
                Lexicon.DEFAULT_A: ("VEC4", {
                    "default": (0,0,0,0), "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "label": ["X", "Y", "Z", "W"]}),
                Lexicon.DEFAULT_B: ("VEC4", {
                    "default": (0,0,0,0), "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "label": ["X", "Y", "Z", "W"]})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[bool]:
        results = []
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        B = parse_param(kw, Lexicon.IN_B, EnumConvertType.ANY, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumBinaryOperation, EnumBinaryOperation.ADD.name)
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType, EnumConvertType.FLOAT.name)
        swap = parse_param(kw, Lexicon.SWAP, EnumConvertType.BOOLEAN, False)
        a_xyzw = parse_param(kw, Lexicon.DEFAULT_A, EnumConvertType.VEC4, (0, 0, 0, 0))
        b_xyzw = parse_param(kw, Lexicon.DEFAULT_B, EnumConvertType.VEC4, (0, 0, 0, 0))
        params = list(zip_longest_fill(A, B, a_xyzw, b_xyzw, op, typ, swap))
        pbar = ProgressBar(len(params))
        for idx, (A, B, a_xyzw, b_xyzw, op, typ, swap) in enumerate(params):
            if not isinstance(A, (list, tuple,)):
                A = [A]
            if not isinstance(B, (list, tuple,)):
                B = [B]
            size = min(3, max(len(A)-1, len(B)-1))
            best_type = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]
            val_a = parse_value(A, best_type, a_xyzw)
            val_a = parse_value(val_a, EnumConvertType.VEC4, a_xyzw)
            val_b = parse_value(B, best_type, b_xyzw)
            val_b = parse_value(val_b, EnumConvertType.VEC4, b_xyzw)

            if swap:
                val_a, val_b = val_b, val_a

            size = max(1, int(typ.value / 10))
            val_a = val_a[:size+1]
            val_b = val_b[:size+1]

            match op:
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
                    val = [a ** b if b >= 0 else 0 for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.MAXIMUM:
                    val = [max(a, val_b[i]) for i, a in enumerate(val_a)]
                case EnumBinaryOperation.MINIMUM:
                    # val = min(val_a, val_b)
                    val = [min(a, val_b[i]) for i, a in enumerate(val_a)]

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
                    val = [int(a) << int(b) if b >= 0 else 0 for a, b in zip(val_a, val_b)]
                case EnumBinaryOperation.BIT_RSHIFT:
                    val = [int(a) >> int(b) if b >= 0 else 0 for a, b in zip(val_a, val_b)]

                # GROUP
                case EnumBinaryOperation.UNION:
                    val = list(set(val_a) | set(val_b))
                case EnumBinaryOperation.INTERSECTION:
                    val = list(set(val_a) & set(val_b))
                case EnumBinaryOperation.DIFFERENCE:
                    val = list(set(val_a) - set(val_b))

                # WEIRD
                case EnumBinaryOperation.BASE:
                    val = list(set(val_a) - set(val_b))

            # cast into correct type....
            default = val
            if len(val) == 0:
                default = [0]

            val = parse_value(val, typ, default)
            results.append(val)
            pbar.update_absolute(idx)
        return (results,)

class StringerNode(CozyBaseNode):
    NAME = "STRINGER (JOV) 🪀"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("STRING", "COUNT",)
    OUTPUT_IS_LIST = (True, False,)
    DESCRIPTION = """
Manipulate strings through filtering
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                # split, join, replace, trim/lift
                Lexicon.FUNCTION: (EnumConvertString._member_names_, {
                    "default": EnumConvertString.SPLIT.name}),
                Lexicon.KEY: ("STRING", {
                    "default":"", "dynamicPrompt":False,
                    "tooltip": "Delimiter (SPLIT/JOIN) or string to use as search string (FIND/REPLACE)."}),
                Lexicon.REPLACE: ("STRING", {
                    "default":"", "dynamicPrompt":False}),
                Lexicon.RANGE: ("VEC3", {
                    "default":(0, -1, 1), "int": True,
                    "tooltip": "Start, End and Step. Values will clip to the actual list size(s)."}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[TensorType, ...]:
        # turn any all inputs into the
        data_list = parse_dynamic(kw, Lexicon.STRING, EnumConvertType.ANY, "")
        if data_list is None:
            logger.warn("no data for list")
            return ([], 0)

        op = parse_param(kw, Lexicon.FUNCTION, EnumConvertString, EnumConvertString.SPLIT.name)[0]
        key = parse_param(kw, Lexicon.KEY, EnumConvertType.STRING, "")[0]
        replace = parse_param(kw, Lexicon.REPLACE, EnumConvertType.STRING, "")[0]
        stenst = parse_param(kw, Lexicon.RANGE, EnumConvertType.VEC3INT, (0, -1, 1))[0]
        results = []
        match op:
            case EnumConvertString.SPLIT:
                results = data_list
                if key != "":
                    results = []
                    for d in data_list:
                        d = [key if len(r) == 0 else r for r in d.split(key)]
                        results.extend(d)
            case EnumConvertString.JOIN:
                results = [key.join(data_list)]
            case EnumConvertString.FIND:
                results = [r for r in data_list if r.find(key) > -1]
            case EnumConvertString.REPLACE:
                results = data_list
                if key != "":
                    results = [r.replace(key, replace) for r in data_list]
            case EnumConvertString.SLICE:
                start, end, step = stenst
                for x in data_list:
                    start = len(x) if start < 0 else min(max(0, start), len(x))
                    end = len(x) if end < 0 else min(max(0, end), len(x))
                    if step != 0:
                        results.append(x[start:end:step])
                    else:
                        results.append(x)
        return (results, len(results),)

class SwizzleNode(CozyBaseNode):
    NAME = "SWIZZLE (JOV) 😵"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY,)
    RETURN_NAMES = ("❔",)
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = """
Swap components between two vectors based on specified swizzle patterns and values. It provides flexibility in rearranging vector elements dynamically.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        names_convert = EnumConvertType._member_names_[3:6]
        d = deep_merge(d, {
            "optional": {
                Lexicon.IN_A: (COZY_TYPE_NUMERICAL, {}),
                Lexicon.IN_B: (COZY_TYPE_NUMERICAL, {}),
                Lexicon.TYPE: (names_convert, {
                    "default": names_convert[0]}),
                Lexicon.SWAP_X: (EnumSwizzle._member_names_, {
                    "default": EnumSwizzle.A_X.name,}),
                Lexicon.SWAP_Y: (EnumSwizzle._member_names_, {
                    "default": EnumSwizzle.A_Y.name,}),
                Lexicon.SWAP_Z: (EnumSwizzle._member_names_, {
                    "default": EnumSwizzle.A_Z.name,}),
                Lexicon.SWAP_W: (EnumSwizzle._member_names_, {
                    "default": EnumSwizzle.A_W.name,}),
                Lexicon.DEFAULT: ("VEC4", {
                    "default": (0,0,0,0), "mij": -sys.float_info.max, "maj": sys.float_info.max})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[float, ...]:
        pA = parse_param(kw, Lexicon.IN_A, EnumConvertType.LIST, None)
        pB = parse_param(kw, Lexicon.IN_B, EnumConvertType.LIST, None)
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType, EnumConvertType.VEC2.name)
        swap_x = parse_param(kw, Lexicon.SWAP_X, EnumSwizzle, EnumSwizzle.A_X.name)
        swap_y = parse_param(kw, Lexicon.SWAP_Y, EnumSwizzle, EnumSwizzle.A_Y.name)
        swap_z = parse_param(kw, Lexicon.SWAP_Z, EnumSwizzle, EnumSwizzle.A_W.name)
        swap_w = parse_param(kw, Lexicon.SWAP_W, EnumSwizzle, EnumSwizzle.A_Z.name)
        default = parse_param(kw, Lexicon.DEFAULT, EnumConvertType.VEC4, (0, 0, 0, 0))
        params = list(zip_longest_fill(pA, pB, typ, swap_x, swap_y, swap_z, swap_w, default))
        results = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, typ, swap_x, swap_y, swap_z, swap_w, default) in enumerate(params):
            default = list(default)
            pA = pA + default[len(pA):]
            pB = pB + default[len(pB):]
            val = vector_swap(pA, pB, swap_x, swap_y, swap_z, swap_w, default)
            val = parse_value(val, typ, val)
            results.append(val)
            pbar.update_absolute(idx)
        return (results,)
