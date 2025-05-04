""" Jovimetrix - Calculation """

import sys
import math
import struct
from enum import Enum
from typing import Any, List
from collections import Counter

import torch
import numpy as np
from scipy.special import gamma

from comfy.utils import ProgressBar

from cozy_comfyui import \
    logger, \
    TensorType, InputType, EnumConvertType, \
    deep_merge, parse_dynamic, parse_param, parse_value, zip_longest_fill

from cozy_comfyui.node import \
    COZY_TYPE_ANY, COZY_TYPE_NUMERICAL, COZY_TYPE_FULL, \
    CozyBaseNode

from ..sup.anim import \
    EnumEase, \
    ease_op

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

class EnumNumberType(Enum):
    INT = 0
    FLOAT = 10

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
    EnumUnaryOperation.FACTORIAL: lambda x: math.factorial(math.abs(int(x))),
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

def to_bits(value):
    if isinstance(value, int):
        return bin(value)[2:]
    elif isinstance(value, float):
        packed = struct.pack('>d', value)
        return ''.join(f'{byte:08b}' for byte in packed)
    elif isinstance(value, str):
        return ''.join(f'{ord(c):08b}' for c in value)
    else:
        raise TypeError(f"Unsupported type: {type(value)}")

def vector_swap(pA: Any, pB: Any, swap_x: EnumSwizzle, x:float, swap_y:EnumSwizzle, y:float,
                swap_z:EnumSwizzle, z:float, swap_w:EnumSwizzle, w:float) -> List[float]:
    """Swap out a vector's values with another vector's values, or a constant fill."""
    def parse(target, targetB, swap, val) -> float:
        if swap == EnumSwizzle.CONSTANT:
            return val
        if swap in [EnumSwizzle.B_X, EnumSwizzle.B_Y, EnumSwizzle.B_Z, EnumSwizzle.B_W]:
            target = targetB
        swap = int(swap.value / 10)
        return target[swap] if swap < len(target) else 0

    return [
        parse(pA, pB, swap_x, x),
        parse(pA, pB, swap_y, y),
        parse(pA, pB, swap_z, z),
        parse(pA, pB, swap_w, w)
    ]

# ==============================================================================
# === CLASS ===
# ==============================================================================

class BitSplitNode(CozyBaseNode):
    NAME = "BIT SPLIT (JOV) ⭄"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, "BOOLEAN",)
    RETURN_NAMES = ("BIT", "BOOL",)
    OUTPUT_TOOLTIPS = (
        "Bits as Numerical output (0 or 1)",
        "Bits as Boolean output (True or False)"
    )
    SORT = 10
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
                "VALUE": (COZY_TYPE_FULL, {"default": None, "tooltip":"the value to convert into bits"}),
                "BITS": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip":"number of output bits requested"}),
                "MSB": ("BOOLEAN", {"default": False, "tooltip":"return the most signifigant bits (True) or least signifigant bits first"})
            }
        })
        return d

    def run(self, **kw) -> tuple[List[int], List[bool]]:
        value = parse_param(kw, "VALUE", EnumConvertType.ANY, 0)
        bits = parse_param(kw, "BITS", EnumConvertType.INT, 8, 1, 64)
        msb = parse_param(kw, "MSB", EnumConvertType.INT, False)
        params = list(zip_longest_fill(value, bits))
        pbar = ProgressBar(len(params))
        results = []
        for idx, (value, bits) in enumerate(params):
            bit_repr = to_bits(value)
            if len(bit_repr) > bits:
                if msb:
                    bit_repr = bit_repr[bits]
                else:
                    bit_repr = bit_repr[-bits:]
            elif msb:
                bit_repr = bit_repr.zfill(bits)
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
    OUTPUT_TOOLTIPS = (
        "Outputs the input at PASS or FAIL depending the evaluation",
        "The comparison result value"
    )
    SORT = 130
    DESCRIPTION = """
Evaluates two inputs (A and B) with a specified comparison operators and optional values for successful and failed comparisons. The node performs the specified operation element-wise between corresponding elements of A and B. If the comparison is successful for all elements, it returns the success value; otherwise, it returns the failure value. The node supports various comparison operators such as EQUAL, GREATER_THAN, LESS_THAN, AND, OR, IS, IN, etc.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "A": (COZY_TYPE_FULL, {
                    "default": 0,
                    "tooltip":"First value to compare"}),
                "B": (COZY_TYPE_FULL, {
                    "default": 0,
                    "tooltip":"Second value to compare"}),
                "PASS": (COZY_TYPE_ANY, {
                    "default": 0,
                    "tooltip": "Passed to OUT on a successful condition"}),
                "FAIL": (COZY_TYPE_ANY, {
                    "default": 0,
                    "tooltip": "Passed to OUT on a failure condition"}),
                "COMPARE": (EnumComparison._member_names_, {
                    "default": EnumComparison.EQUAL.name,
                    "tooltip": "Comparison function. Sends the data in PASS on successful comparison to OUT, otherwise sends the value in FAIL"}),
                "FLIP": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the inputs A and B"}),
                "INVERT": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the successful and failure inputs"}),
            }
        })
        return d

    def run(self, **kw) -> tuple[Any, Any]:
        A = parse_param(kw, "A", EnumConvertType.ANY, 0)
        B = parse_param(kw, "B", EnumConvertType.ANY, 0)
        size = max(len(A), len(B))
        good = parse_param(kw, "PASS", EnumConvertType.ANY, 0)[:size]
        fail = parse_param(kw, "FAIL", EnumConvertType.ANY, 0)[:size]
        op = parse_param(kw, "COMPARE", EnumComparison, EnumComparison.EQUAL.name)[:size]
        flip = parse_param(kw, "FLIP", EnumConvertType.BOOLEAN, False)[:size]
        invert = parse_param(kw, "INVERT", EnumConvertType.BOOLEAN, False)[:size]
        params = list(zip_longest_fill(A, B, good, fail, op, flip, invert))
        pbar = ProgressBar(len(params))
        vals = []
        results = []
        for idx, (A, B, good, fail, op, flip, invert) in enumerate(params):
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

            if flip:
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
        else:
            outs = list(outs)
        return outs, *vals,

class LerpNode(CozyBaseNode):
    NAME = "LERP (JOV) 🔰"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY,)
    RETURN_NAMES = ("🦄",)
    OUTPUT_TOOLTIPS = (
        f"Output can vary depending on the type chosen in the {"TYPE"} parameter"
    )
    SORT = 30
    DESCRIPTION = """
Calculate linear interpolation between two values or vectors based on a blending factor (alpha).

The node accepts optional start (IN_A) and end (IN_B) points, a blending factor (FLOAT), and various input types for both start and end points, such as single values (X, Y), 2-value vectors (IN_A2, IN_B2), 3-value vectors (IN_A3, IN_B3), and 4-value vectors (IN_A4, IN_B4).

Additionally, you can specify the easing function (EASE) and the desired output type (TYPE). It supports various easing functions for smoother transitions.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        names_convert = EnumConvertType._member_names_[:6]
        d = deep_merge(d, {
            "optional": {
                "A": (COZY_TYPE_FULL, {
                    "tooltip": "Custom Start Point"
                }),
                "B": (COZY_TYPE_FULL, {
                    "tooltip": "Custom End Point"
                }),
                "ALPHA": ("VEC4", {
                    "default": (0.5, 0.5, 0.5, 0.5), "mij": 0., "maj": 1.0,
                    "tooltip": "Blend Amount. 0 = full A, 1 = full B"
                }),
                "AA": ("VEC4", {
                    "default": (0, 0, 0, 0),
                    "tooltip":"default value vector for A"
                }),
                "BB": ("VEC4", {
                    "default": (1,1,1,1),
                    "tooltip":"default value vector for B"
                }),
                "TYPE": (names_convert, {
                    "default": "FLOAT",
                    "tooltip":"Output type desired from resultant operation"
                }),
                "EASE": (["NONE"] + EnumEase._member_names_, {
                    "default": "NONE"
                }),
            }
        })
        return d

    def run(self, **kw) -> tuple[Any, Any]:
        A = parse_param(kw, "A", EnumConvertType.ANY, 0)
        B = parse_param(kw, "B", EnumConvertType.ANY, 0)
        a_xyzw = parse_param(kw, "AA", EnumConvertType.VEC4, (0, 0, 0, 0))
        b_xyzw = parse_param(kw, "BB", EnumConvertType.VEC4, (1, 1, 1, 1))
        alpha = parse_param(kw, "FLOAT",EnumConvertType.VEC4, (0.5,0.5,0.5,0.5), 0, 1)
        op = parse_param(kw, "EASE", EnumEase, EnumEase.SIN_IN_OUT.name)
        typ = parse_param(kw, "TYPE", EnumNumberType, EnumNumberType.FLOAT.name)
        values = []
        params = list(zip_longest_fill(A, B, a_xyzw, b_xyzw, alpha, op, typ))
        pbar = ProgressBar(len(params))
        for idx, (A, B, a_xyzw, b_xyzw, alpha, op, typ) in enumerate(params):
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

            # logger.debug([A, B, val_a, val_b, alpha, size])

            if op == "NONE":
                val = [val_b[x] * alpha[x] + val_a[x] * (1 - alpha[x]) for x in range(size)]
            else:
                # ease = EnumEase[op]
                val = [ease_op(op, val_a[x], val_b[x], alpha=alpha[x]) for x in range(size)]

            convert = int if "INT" in typ.name else float
            ret = []
            for v in val:
                try:
                    ret.append(convert(v))
                except OverflowError:
                    ret.append(0)
                except Exception as e:
                    logger.error(f"{e} :: {op}")
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
    OUTPUT_TOOLTIPS = (
        "Output type will match the input type"
    )
    SORT = 10
    DESCRIPTION = """
Perform single function operations like absolute value, mean, median, mode, magnitude, normalization, maximum, or minimum on input values.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "A": (COZY_TYPE_NUMERICAL, {
                    "default": None}),
                "FUNCTION": (EnumUnaryOperation._member_names_, {
                    "default": EnumUnaryOperation.ABS.name})
            }
        })
        return d

    def run(self, **kw) -> tuple[bool]:
        results = []
        A = parse_param(kw, "A", EnumConvertType.ANY, 0)
        op = parse_param(kw, "FUNCTION", EnumUnaryOperation, EnumUnaryOperation.ABS.name)
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
            elif isinstance(A, (TensorType,)):
                typ = EnumConvertType.IMAGE

            val = parse_value(A, typ, 0)
            if not isinstance(val, (list, tuple, )):
                val = [val]
            val = [float(v) for v in val]
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

            convert = int if isinstance(A, (bool, int, np.uint8, np.uint16, np.uint32, np.uint64)) else float
            ret = []
            for v in val:
                try:
                    ret.append(convert(v))
                except OverflowError:
                    ret.append(0)
                except Exception as e:
                    logger.error(f"{e} :: {op}")
                    ret.append(0)
            val = parse_value(val, typ, 0)
            results.append(val)
            pbar.update_absolute(idx)
        return (results,)

class OPBinaryNode(CozyBaseNode):
    NAME = "OP BINARY (JOV) 🌟"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY,)
    RETURN_NAMES = ("❔",)
    OUTPUT_TOOLTIPS = (
        "Output type will match the input type"
    )

    SORT = 20
    DESCRIPTION = """
Execute binary operations like addition, subtraction, multiplication, division, and bitwise operations on input values, supporting various data types and vector sizes.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        names_convert = EnumConvertType._member_names_[:6]
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "A": (COZY_TYPE_NUMERICAL, {
                    "default": None,
                    "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
                "B": (COZY_TYPE_NUMERICAL, {
                    "default": None,
                    "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
                "FUNCTION": (EnumBinaryOperation._member_names_, {
                    "default": EnumBinaryOperation.ADD.name,
                    "tooltip":"Arithmetic operation to perform"}),
                "TYPE": (names_convert, {
                    "default": names_convert[2],
                    "tooltip":"Output type desired from resultant operation"}),
                "FLIP": ("BOOLEAN", {
                    "default": False}),
                "AA": ("VEC4", {
                    "default": (0,0,0,0),
                    "label": ["X", "Y", "Z", "W"],
                    "tooltip":"value vector"}),
                "BB": ("VEC4", {
                    "default": (0,0,0,0),
                    "label": ["X", "Y", "Z", "W"],
                    "tooltip":"value vector"}),
            }
        })
        return d

    def run(self, **kw) -> tuple[bool]:
        results = []
        A = parse_param(kw, "A", EnumConvertType.ANY, None)
        B = parse_param(kw, "B", EnumConvertType.ANY, None)
        a_xyzw = parse_param(kw, "AA", EnumConvertType.VEC4, (0, 0, 0, 0))
        b_xyzw = parse_param(kw, "BB", EnumConvertType.VEC4, (0, 0, 0, 0))
        op = parse_param(kw, "FUNCTION", EnumBinaryOperation, EnumBinaryOperation.ADD.name)
        typ = parse_param(kw, "TYPE", EnumConvertType, EnumConvertType.FLOAT.name)
        flip = parse_param(kw, "FLIP", EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(A, B, a_xyzw, b_xyzw, op, typ, flip))
        pbar = ProgressBar(len(params))
        for idx, (A, B, a_xyzw, b_xyzw, op, typ, flip) in enumerate(params):
            size = min(3, max(0 if not isinstance(A, (list,)) else len(A), 0 if not isinstance(B, (list,)) else len(B)))
            best_type = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]
            print(type(A), type(B), A, B, a_xyzw)
            val_a = parse_value(A, best_type, a_xyzw)
            print(val_a)
            return
            val_a = parse_value(val_a, EnumConvertType.VEC4, a_xyzw)
            val_b = parse_value(B, best_type, b_xyzw)
            val_b = parse_value(val_b, EnumConvertType.VEC4, b_xyzw)

            print(val_a, val_b)

            #val_a = parse_value(A, EnumConvertType.VEC4, A if A is not None else a_xyzw)
            #val_b = parse_value(B, EnumConvertType.VEC4, B if B is not None else b_xyzw)

            if flip:
                val_a, val_b = val_b, val_a
            #size = max(1, int(typ.value / 10))
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
        return results

class StringerNode(CozyBaseNode):
    NAME = "STRINGER (JOV) 🪀"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("STRING", "COUNT",)
    SORT = 44
    DESCRIPTION = """
Manipulate strings through filtering
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                # split, join, replace, trim/lift
                "FUNCTION": (EnumConvertString._member_names_, {
                    "default": EnumConvertString.SPLIT.name,
                    "tooltip":"Operation to perform on the input string"}),
                "KEY": ("STRING", {
                    "default":"", "dynamicPrompt":False,
                    "tooltip":"Delimiter (SPLIT/JOIN) or string to use as search string (FIND/REPLACE)."}),
                "REPLACE": ("STRING", {
                    "default":"", "dynamicPrompt":False}),
                "RANGE": ("VEC3", {
                    "default":(0, -1, 1),
                    "tooltip":"Start, End and Step. Values will clip to the actual list size(s)."}),
            }
        })
        return d

    def run(self, **kw) -> tuple[TensorType, ...]:
        # turn any all inputs into the
        data_list = parse_dynamic(kw, "❔", EnumConvertType.ANY, "")
        if data_list is None:
            logger.warn("no data for list")
            return ([],)
        # flat list of ALL the dynamic inputs...
        #data_list = flatten(data_list)
        # single operation mode -- like array node
        op = parse_param(kw, "FUNCTION", EnumConvertString, EnumConvertString.SPLIT.name)[0]
        key = parse_param(kw, "KEY", EnumConvertType.STRING, "")[0]
        replace = parse_param(kw, "REPLACE", EnumConvertType.STRING, "")[0]
        stenst = parse_param(kw, "RANGE", EnumConvertType.VEC3INT, (0, -1, 1))[0]
        results = []
        match op:
            case EnumConvertString.SPLIT:
                results = data_list
                if key != "":
                    results = [r.split(key) for r in data_list]
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
        if len(results) == 0:
            results = [""]
        return (results, [len(r) for r in results],) if len(results) > 1 else (results[0], len(results[0]),)

class SwizzleNode(CozyBaseNode):
    NAME = "SWIZZLE (JOV) 😵"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY,)
    RETURN_NAMES = ("🦄",)
    SORT = 40
    DESCRIPTION = """
Swap components between two vectors based on specified swizzle patterns and values. It provides flexibility in rearranging vector elements dynamically.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        names_convert = EnumConvertType._member_names_[3:6]
        d = deep_merge(d, {
            "optional": {
                "A": (COZY_TYPE_NUMERICAL, {}),
                "B": (COZY_TYPE_NUMERICAL, {}),
                "TYPE": (names_convert, {
                    "default": names_convert[2],
                    "tooltip":"Output type desired from resultant operation"
                }),
                "SWAP_X": (EnumSwizzle._member_names_, {
                    "default": EnumSwizzle.A_X.name,
                    "tooltip": "Replace input Red channel with target channel or constant"
                }),
                "SWAP_Y": (EnumSwizzle._member_names_, {
                    "default": EnumSwizzle.A_Y.name,
                    "tooltip": "Replace input Green channel with target channel or constant"
                }),
                "SWAP_Z": (EnumSwizzle._member_names_, {
                    "default": EnumSwizzle.A_Z.name,
                    "tooltip": "Replace input Blue channel with target channel or constant"
                }),
                "SWAP_W": (EnumSwizzle._member_names_, {
                    "default": EnumSwizzle.A_W.name,
                    "tooltip": "Replace input W channel with target channel or constant"
                }),
                "VEC": ("VEC4", {
                    "default": (0,0,0,0), "mij": -sys.maxsize, "maj": sys.maxsize,
                    "tooltip": "Default values for missing channels"
                })
            }
        })
        return d

    def run(self, **kw) -> tuple[TensorType, ...]:
        pA = parse_param(kw, "A", EnumConvertType.VEC4, (0,0,0,0))
        pB = parse_param(kw, "B", EnumConvertType.VEC4, (0,0,0,0))
        swap_x = parse_param(kw, "SWAP_X", EnumSwizzle, EnumSwizzle.A_X.name)
        swap_y = parse_param(kw, "SWAP_Y", EnumSwizzle, EnumSwizzle.A_Y.name)
        swap_z = parse_param(kw, "SWAP_Z", EnumSwizzle, EnumSwizzle.A_W.name)
        swap_w = parse_param(kw, "SWAP_W", EnumSwizzle, EnumSwizzle.A_Z.name)
        default = parse_param(kw, "VEC", EnumConvertType.VEC4, 0, -sys.maxsize, sys.maxsize)

        params = list(zip_longest_fill(pA, pB, swap_x, x, swap_y, y, swap_z, z, swap_w, w))
        results = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, swap_x, x, swap_y, y, swap_z, z, swap_w, w) in enumerate(params):
            val = vector_swap(pA, pB, swap_x, x, swap_y, y, swap_z, z, swap_w, w)
            results.append(val)
            pbar.update_absolute(idx)
        return results
