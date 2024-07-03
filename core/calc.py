"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Calculation
"""

import os
import sys
import math
import random
from enum import Enum
from typing import Any, Tuple
from collections import Counter

import torch
import numpy as np
from scipy.special import gamma
from loguru import logger

from comfy.utils import ProgressBar
from nodes import interrupt_processing

from Jovimetrix import JOV_TYPE_FULL, JOV_TYPE_COMFY, JOV_TYPE_IMAGE, JOV_TYPE_NUMBER, JOV_TYPE_VECTOR, comfy_message, parse_reset, JOVBaseNode, ComfyAPIMessage, \
    TimedOutException, JOV_TYPE_ANY
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_param, parse_value, vector_swap, \
    zip_longest_fill, EnumConvertType, EnumSwizzle
from Jovimetrix.sup.anim import ease_op, wave_op, EnumWave, EnumEase
from Jovimetrix.sup.util import parse_param, zip_longest_fill, EnumConvertType

# =============================================================================

JOV_CATEGORY = "CALC"

# min amount of time before showing the cancel dialog
JOV_DELAY_MIN = 5
try: JOV_DELAY_MIN = int(os.getenv("JOV_DELAY_MIN", JOV_DELAY_MIN))
except: pass
JOV_DELAY_MIN = max(1, JOV_DELAY_MIN)

# max 10 minutes to start
JOV_DELAY_MAX = 600
try: JOV_DELAY_MAX = int(os.getenv("JOV_DELAY_MAX", JOV_DELAY_MAX))
except: pass

# =============================================================================

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

class Results(object):
    def __init__(self, *arg, **kw) -> None:
        self.frame = []
        self.lin = []
        self.fixed = []
        self.trigger = []

# =============================================================================

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

# =============================================================================

class CalcUnaryOPNode(JOVBaseNode):
    NAME = "OP UNARY (JOV) 🎲"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_NUMBER,)
    RETURN_NAMES = (Lexicon.UNKNOWN,)
    SORT = 10
    DESCRIPTION = """
The Unary Operation node performs unary operations like absolute value, mean, median, mode, magnitude, normalization, maximum, or minimum on input values.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.IN_A: (JOV_TYPE_FULL, {"default": None}),
                Lexicon.FUNC: (EnumUnaryOperation._member_names_, {"default": EnumUnaryOperation.ABS.name})
            },
            "outputs": {
                0: (Lexicon.UNKNOWN, {"tooltip":"Output type will match the input type"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[bool]:
        results = []
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        op = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumUnaryOperation.ABS.name)
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
            if not isinstance(val, (list, tuple, )):
                val = [val]
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

class CalcBinaryOPNode(JOVBaseNode):
    NAME = "OP BINARY (JOV) 🌟"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_NUMBER,)
    RETURN_NAMES = (Lexicon.UNKNOWN,)
    SORT = 20
    DESCRIPTION = """
The Binary Operation node executes binary operations like addition, subtraction, multiplication, division, and bitwise operations on input values, supporting various data types and vector sizes.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        names_convert = EnumConvertType._member_names_[:10]
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.IN_A: (JOV_TYPE_FULL, {"default": None,
                                        "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
                Lexicon.IN_B: (JOV_TYPE_FULL, {"default": None,
                                        "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
                Lexicon.FUNC: (EnumBinaryOperation._member_names_, {"default": EnumBinaryOperation.ADD.name, "tooltip":"Arithmetic operation to perform"}),
                Lexicon.TYPE: (names_convert, {"default": names_convert[2],
                                            "tooltip":"Output type desired from resultant operation"}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
                Lexicon.IN_A+Lexicon.IN_A: ("VEC4", {"default": (0,0,0,0),
                                        "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
                                        "tooltip":"value vector"}),
                Lexicon.IN_B+Lexicon.IN_B: ("VEC4", {"default": (0,0,0,0),
                                        "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
                                        "tooltip":"value vector"}),
            },
            "outputs": {
                0: (Lexicon.UNKNOWN, {"tooltip":"Output type will match the input type"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[bool]:
        results = []
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        B = parse_param(kw, Lexicon.IN_B, EnumConvertType.ANY, None)
        a_xyzw = parse_param(kw, Lexicon.IN_A+Lexicon.IN_A, EnumConvertType.VEC4, [(0, 0, 0, 0)])
        b_xyzw = parse_param(kw, Lexicon.IN_B+Lexicon.IN_B, EnumConvertType.VEC4, [(0, 0, 0, 0)])
        op = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumBinaryOperation.ADD.name)
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumConvertType.FLOAT.name)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(A, B, a_xyzw, b_xyzw, op, typ, flip))
        pbar = ProgressBar(len(params))
        for idx, (A, B, a_xyzw, b_xyzw, op, typ, flip) in enumerate(params):
            logger.debug(f'val {A}, {B}, {a_xyzw}, {b_xyzw}')
            # [[0.008793391303918097, 0.008793391303918097, 0.008793391303918097]], [[0.008793391303918097, 0.008793391303918097, 0.008793391303918097]], (0, 0, 0, 0), (0, 0, 0, 0)
            typ = EnumConvertType[typ]

            size = min(3, max(0 if not isinstance(A, (list,)) else len(A), 0 if not isinstance(B, (list,)) else len(B)))
            best_type = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]
            val_a = parse_value(A, best_type, a_xyzw)
            val_a = parse_value(val_a, EnumConvertType.VEC4, a_xyzw)
            val_b = parse_value(B, best_type, b_xyzw)
            val_b = parse_value(val_b, EnumConvertType.VEC4, b_xyzw)

            val_a = parse_value(A, EnumConvertType.VEC4, A if A is not None else a_xyzw)
            val_b = parse_value(B, EnumConvertType.VEC4, B if B is not None else b_xyzw)
            logger.debug(f'val {val_a}, {val_b}')
            # (0, 0, 0, 0), (0, 0, 0, 0)
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

            # cast into correct type....
            default = val
            if len(val) == 0:
                default = [0]
            val = parse_value(val, typ, default)
            results.append(val)
            pbar.update_absolute(idx)
        return results

class ComparisonNode(JOVBaseNode):
    NAME = "COMPARISON (JOV) 🕵🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_NUMBER, JOV_TYPE_NUMBER,)
    RETURN_NAMES = (Lexicon.TRIGGER, Lexicon.VALUE,)
    SORT = 130
    DESCRIPTION = """
The Comparison node evaluates two inputs based on a specified operation. It accepts two inputs (A and B), comparison operators, and optional values for successful and failed comparisons. The node performs the specified operation element-wise between corresponding elements of A and B. If the comparison is successful for all elements, it returns the success value; otherwise, it returns the failure value. The node supports various comparison operators such as EQUAL, GREATER_THAN, LESS_THAN, AND, OR, IS, IN, etc.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.IN_A: (JOV_TYPE_FULL, {"default": None, "tooltip":"Master Comparator"}),
                Lexicon.IN_B: (JOV_TYPE_FULL, {"default": None, "tooltip":"Secondary Comparator"}),
                Lexicon.COMP_A: (JOV_TYPE_ANY, {"default": None}),
                Lexicon.COMP_B: (JOV_TYPE_ANY, {"default": None}),
                Lexicon.COMPARE: (EnumComparison._member_names_, {"default": EnumComparison.EQUAL.name}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
            },
            "outputs": {
                0: (Lexicon.TRIGGER, {"tooltip":f"Outputs the input at {Lexicon.IN_A} or {Lexicon.IN_B} depending on which evaluated `TRUE`"}),
                1: (Lexicon.VALUE, {"tooltip":"The comparison result value"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[Any, Any]:
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        B = parse_param(kw, Lexicon.IN_B, EnumConvertType.ANY, None)
        good = parse_param(kw, Lexicon.COMP_A, EnumConvertType.ANY, None)
        fail = parse_param(kw, Lexicon.COMP_B, EnumConvertType.ANY, None)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        op = parse_param(kw, Lexicon.COMPARE, EnumConvertType.STRING, EnumComparison.EQUAL.name)
        params = list(zip_longest_fill(A, B, good, fail, flip, op))
        pbar = ProgressBar(len(params))
        vals = []
        results = []
        for idx, (A, B, good, fail, flip, op) in enumerate(params):
            if not isinstance(A, (list, set, tuple)):
                A = [A]
            if not isinstance(B, (list, set, tuple)):
                B = [B]
            size = min(4, max(len(A), len(B))) - 1
            typ = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]
            val_a = parse_value(A, typ, [A[-1]] * size)
            val_b = parse_value(B, typ, [B[-1]] * size)
            if flip:
                val_a, val_b = val_b, val_a

            if not isinstance(val_a, (list, tuple)):
                val_a = [val_a]
            if not isinstance(val_b, (list, tuple)):
                val_b = [val_b]

            op = EnumComparison[op]
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

            output = good if all([bool(v) for v in val]) else fail
            results.append([output, val])
            pbar.update_absolute(idx)

        outs, vals = zip(*results)
        if isinstance(outs[0], (torch.Tensor,)):
            if len(outs) > 1:
                outs = torch.cat(outs, dim=0)
            else:
                outs = outs[0].unsqueeze(0)
        else:
            outs = list(outs)
        return outs, list(vals)

class DelayNode(JOVBaseNode):
    NAME = "DELAY (JOV) ✋🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_ANY,)
    RETURN_NAMES = (Lexicon.PASS_OUT,)
    SORT = 240
    DESCRIPTION = """
Delay node used to introduce pauses in the workflow. It accepts an optional input to pass through and a timer parameter to specify the duration of the delay. If no timer is provided, it defaults to a maximum delay. During the delay, it periodically checks for messages to interrupt the delay. Once the delay is completed, it returns the input passed to it.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PASS_IN: (JOV_TYPE_ANY, {"default": None}),
                Lexicon.TIMER: ("INT", {"step": 1, "default" : 0, "min": -1}),
            },
            "outputs": {
                0: (Lexicon.PASS_OUT, {"tooltip":f"Pass through data when the delay ends"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, ident, **kw) -> Tuple[Any]:
        delay = parse_param(kw, Lexicon.TIMER, EnumConvertType.INT, -1, 0, JOV_DELAY_MAX)[0]
        if delay < 0:
            delay = JOV_DELAY_MAX
        if delay > JOV_DELAY_MIN:
            comfy_message(ident, "jovi-delay-user", {"id": ident, "timeout": delay})
        step = 1
        pbar = ProgressBar(delay)
        while step <= delay:
            # comfy_message(ident, "jovi-delay-update", {"id": ident, "timeout": step})
            try:
                data = ComfyAPIMessage.poll(ident, timeout=1)
                if data.get('cancel', True):
                    interrupt_processing(True)
                    logger.warning(f"delay [cancelled] ({step}): {ident}")
                    break
            except TimedOutException as _:
                if step % 10 == 0:
                    logger.info(f"delay [continue] ({step}): {ident}")
            pbar.update_absolute(step)
            step += 1
        return kw[Lexicon.PASS_IN],

class LerpNode(JOVBaseNode):
    NAME = "LERP (JOV) 🔰"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_FULL,)
    RETURN_NAMES = (Lexicon.ANY_OUT,)
    SORT = 30
    DESCRIPTION = """
The Lerp Node calculates linear interpolation between two values or vectors based on a blending factor (alpha). The node accepts optional start (IN_A) and end (IN_B) points, a blending factor (FLOAT), and various input types for both start and end points, such as single values (X, Y), 2-value vectors (IN_A2, IN_B2), 3-value vectors (IN_A3, IN_B3), and 4-value vectors (IN_A4, IN_B4). Additionally, you can specify the easing function (EASE) and the desired output type (TYPE). It supports various easing functions for smoother transitions.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        names_convert = EnumConvertType._member_names_[:10]
        d.update({
            "optional": {
                Lexicon.IN_A: (JOV_TYPE_FULL, {"tooltip": "Custom Start Point"}),
                Lexicon.IN_B: (JOV_TYPE_FULL, {"tooltip": "Custom End Point"}),
                Lexicon.FLOAT: ("FLOAT", {"default": 0.5, "min": 0., "max": 1.0,
                                        "step": 0.001, "precision": 4, "round": 0.00001,
                                        "tooltip": "Blend Amount. 0 = full A, 1 = full B"}),
                Lexicon.EASE: (["NONE"] + EnumEase._member_names_, {"default": "NONE"}),
                Lexicon.TYPE: (names_convert, {"default": names_convert[2],
                                            "tooltip":"Output type desired from resultant operation"}),
                Lexicon.IN_A+Lexicon.IN_A: ("VEC4", {"default": (0, 0, 0, 0),
                                        #"min": -sys.maxsize, "max": sys.maxsize,
                                        "label": [Lexicon.X, Lexicon.Y],
                                        "tooltip":"default value vector for A"}),
                Lexicon.IN_B+Lexicon.IN_B: ("VEC4", {"default": (1,1,1,1),
                                        #"min": -sys.maxsize, "max": sys.maxsize,
                                        "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
                                        "tooltip":"default value vector for B"}),
            },
            "outputs": {
                0: (Lexicon.ANY_OUT, {"tooltip":f"Output can vary depending on the type chosen in the {Lexicon.TYPE} parameter"}),
                1: (Lexicon.FLOAT, {"tooltip":f"1"}),
                2: (Lexicon.INT, {"tooltip":f"2"}),
                3: (Lexicon.VEC, {"tooltip":f"3"}),
                4: (Lexicon.XY, {"tooltip":f"4"}),
                5: (Lexicon.XYZ, {"tooltip":f"5"}),
                6: (Lexicon.XYZW, {"tooltip":f"6"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[Any, Any]:
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        B = parse_param(kw, Lexicon.IN_B, EnumConvertType.ANY, None)
        a_xyzw = parse_param(kw, Lexicon.IN_A+Lexicon.IN_A, EnumConvertType.VEC4, [(0, 0, 0, 0)])
        b_xyzw = parse_param(kw, Lexicon.IN_B+Lexicon.IN_B, EnumConvertType.VEC4, [(1, 1, 1, 1)])
        alpha = parse_param(kw, Lexicon.FLOAT,EnumConvertType.FLOAT, 0.5, 0, 1)
        op = parse_param(kw, Lexicon.EASE, EnumConvertType.STRING, "NONE")
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumNumberType.FLOAT.name)
        values = []
        params = list(zip_longest_fill(A, B, a_xyzw, b_xyzw, alpha, op, typ))
        pbar = ProgressBar(len(params))
        for idx, (A, B, a_xyzw, b_xyzw, alpha, op, typ) in enumerate(params):
            # make sure we only interpolate between the longest "stride" we can
            size = min(3, max(0 if not isinstance(A, (list,)) else len(A), 0 if not isinstance(B, (list,)) else len(B)))
            best_type = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]
            val_a = parse_value(A, best_type, a_xyzw)
            val_a = parse_value(val_a, EnumConvertType.VEC4, a_xyzw)
            val_b = parse_value(B, best_type, b_xyzw)
            val_b = parse_value(val_b, EnumConvertType.VEC4, b_xyzw)
            alpha = parse_value(alpha, EnumConvertType.VEC4, alpha)
            typ = EnumConvertType[typ]
            size = max(1, int(typ.value / 10))
            if size > 1:
                val_a = val_a[:size]
                val_b = val_b[:size]
            else:
                val_a = [val_a[0]]
                val_b = [val_b[0]]

            if op == "NONE":
                val = [val_b[x] * alpha[x] + val_a[x] * (1 - alpha[x]) for x in range(size)]
            else:
                ease = EnumEase[op]
                val = [ease_op(ease, val_a[x], val_b[x], alpha=alpha[x]) for x in range(size)]

            values.append(val)
            pbar.update_absolute(idx)
        return (values, )

class SwizzleNode(JOVBaseNode):
    NAME = "SWIZZLE (JOV) 😵"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_VECTOR,)
    RETURN_NAMES = (Lexicon.ANY_OUT,)
    SORT = 40
    DESCRIPTION = """
The Swap Node swaps components between two vectors based on specified swizzle patterns and values. It provides flexibility in rearranging vector elements dynamically.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        names_convert = EnumConvertType._member_names_[3:10]
        d.update({
            "optional": {
                Lexicon.IN_A: (JOV_TYPE_VECTOR, {}),
                Lexicon.IN_B: (JOV_TYPE_VECTOR, {}),
                Lexicon.TYPE: (names_convert, {"default": names_convert[2],
                                            "tooltip":"Output type desired from resultant operation"}),
                Lexicon.SWAP_X: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_X.name}),
                Lexicon.X: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
                Lexicon.SWAP_Y: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_Y.name}),
                Lexicon.Y: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
                Lexicon.SWAP_Z: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_Z.name}),
                Lexicon.Z: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize}),
                Lexicon.SWAP_W: (EnumSwizzle._member_names_, {"default": EnumSwizzle.A_W.name}),
                Lexicon.W: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.IN_A, EnumConvertType.VEC4, [(0,0,0,0)])
        pB = parse_param(kw, Lexicon.IN_B, EnumConvertType.VEC4, [(0,0,0,0)])
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
        return results

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) ⏱"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", JOV_TYPE_ANY)
    RETURN_NAMES = (Lexicon.VALUE, Lexicon.LINEAR, Lexicon.FPS, Lexicon.TRIGGER)
    SORT = 50
    DESCRIPTION = """
The `Tick` node acts as a timer and frame counter, emitting pulses or signals based on time intervals or BPM settings. It allows precise synchronization and control over animation sequences, with options to adjust FPS, BPM, and loop points. This node is useful for generating time-based events or driving animations with rhythmic precision.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                # data to pass on a pulse of the loop
                Lexicon.TRIGGER: (JOV_TYPE_ANY, {"default": None,
                                             "tooltip":"Output to send when beat (BPM setting) is hit"}),
                # forces a MOD on CYCLE
                Lexicon.VALUE: ("INT", {"min": 0, "default": 0, "step": 1,
                                        "tooltip": "the current frame number of the tick"}),
                Lexicon.LOOP: ("INT", {"min": 0, "max": 32767, "default": 0, "step": 1,
                                       "tooltip": "number of frames before looping starts. 0 means continuous playback (no loop point)"}),
                #
                Lexicon.FPS: ("INT", {"min": 1, "default": 24, "step": 1,
                                      "tooltip": "Fixed frame step rate based on FPS (1/FPS)"}),
                Lexicon.BPM: ("FLOAT", {"min": 1, "max": 60000, "default": 120, "step": 1,
                                        "tooltip": "BPM trigger rate to send the input. If input is empty, TRUE is sent on trigger"}),
                Lexicon.NOTE: ("INT", {"default": 4, "min": 1, "max": 256, "step": 1,
                                    "tooltip":"Number of beats per measure. Quarter note is 4, Eighth is 8, 16 is 16, etc."}),
                # stick the current "count"
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
                # manual total = 0
                Lexicon.RESET: ("BOOLEAN", {"default": False}),
                # how many frames to dump....
                Lexicon.BATCH: ("INT", {"min": 1, "default": 1, "step": 1, "max": 32767, "tooltip": "Number of frames wanted"}),
                Lexicon.STEP: ("INT", {"default": 0, "step": 1, "tooltip": "Steps/Stride between pulses -- useful to do odd or even batches. If set to 0 will stretch from (VAL -> LOOP) / Batch giving a linear range of values."}),
            }
        })
        return Lexicon._parse(d, cls)

    """
    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")
    """

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        # how many pulses we have done -- total unless reset
        self.__frame = 0

    def run(self, ident, **kw) -> Tuple[int, float, float, Any]:
        passthru = parse_param(kw, Lexicon.TRIGGER, EnumConvertType.ANY, None)[0]
        stride = parse_param(kw, Lexicon.STEP, EnumConvertType.INT, 0, 0)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0)[0]
        self.__frame = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, self.__frame)[0]
        if loop != 0:
            self.__frame %= loop
        # start_frame = max(0, start_frame)
        hold = parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False)[0]
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1)[0]
        bpm = parse_param(kw, Lexicon.BPM, EnumConvertType.INT, 120, 1)[0]
        divisor = parse_param(kw, Lexicon.NOTE, EnumConvertType.INT, 4, 1)[0]
        beat = 60. / max(1., bpm) / divisor
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 1, 1)[0]
        step_fps = 1. / max(1., float(fps))
        reset = parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]
        if loop == 0 and (parse_reset(ident) > 0 or reset):
            self.__frame = 0
        trigger = None
        results = Results()
        pbar = ProgressBar(batch)
        step = stride if stride != 0 else max(1, loop / batch)
        for idx in range(batch):
            trigger = False
            lin = self.__frame if loop == 0 else self.__frame / loop
            fixed_step = math.fmod(self.__frame * step_fps, fps)
            if (math.fmod(fixed_step, beat) == 0):
                trigger = [passthru]
            if loop != 0:
                self.__frame %= loop
            results.frame.append(self.__frame)
            results.lin.append(lin)
            results.fixed.append(float(fixed_step))
            results.trigger.append(trigger)
            if not hold:
                self.__frame += step
            pbar.update_absolute(idx)
        if batch < 2:
            comfy_message(ident, "jovi-tick", {"i": self.__frame})
        return (results.frame, results.lin, results.fixed, results.trigger,)

class ValueNode(JOVBaseNode):
    NAME = "VALUE (JOV) 🧬"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_NUMBER, JOV_TYPE_NUMBER, JOV_TYPE_NUMBER, JOV_TYPE_NUMBER, JOV_TYPE_NUMBER,)
    RETURN_NAMES = (Lexicon.ANY_OUT, Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W)
    SORT = 5
    DESCRIPTION = """
The Value Node supplies raw or default values for various data types, supporting vector input with components for X, Y, Z, and W. It also provides a string input option.
"""
    UPDATE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()

        typ = EnumConvertType._member_names_
        try: typ.pop(typ.index('IMAGE'))
        except: pass
        try: typ.pop(typ.index('LATENT'))
        except: pass
        try: typ.pop(typ.index('ANY'))
        except: pass
        try: typ.pop(typ.index('MASK'))
        except: pass
        d.update({
            "optional": {
                Lexicon.IN_A: (JOV_TYPE_ANY, {"default": None,
                                        "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
                Lexicon.TYPE: (typ, {"default": EnumConvertType.BOOLEAN.name,
                                    "tooltip":"Take the input and convert it into the selected type."}),
                Lexicon.X: (JOV_TYPE_ANY, {"default": 0, "min": -sys.maxsize,
                                    "max": sys.maxsize, "step": 0.01, "precision": 6,
                                    "forceInput": True}),
                Lexicon.Y: (JOV_TYPE_ANY, {"default": 0, "min": -sys.maxsize,
                                    "max": sys.maxsize, "step": 0.01, "precision": 6,
                                    "forceInput": True}),
                Lexicon.Z: (JOV_TYPE_ANY, {"default": 0, "min": -sys.maxsize,
                                    "max": sys.maxsize, "step": 0.01, "precision": 6,
                                    "forceInput": True}),
                Lexicon.W: (JOV_TYPE_ANY, {"default": 0, "min": -sys.maxsize,
                                    "max": sys.maxsize, "step": 0.01, "precision": 6,
                                    "forceInput": True}),
                Lexicon.IN_A+Lexicon.IN_A: ("VEC4", {"default": (0, 0, 0, 0),
                                    #"min": -sys.maxsize, "max": sys.maxsize,
                                    "label": [Lexicon.X, Lexicon.Y],
                                    "tooltip":"default value vector for A"}),
                Lexicon.SEED: ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                Lexicon.IN_B+Lexicon.IN_B: ("VEC4", {"default": (1,1,1,1),
                                    #"min": -sys.maxsize, "max": sys.maxsize,
                                    "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
                                    "tooltip":"default value vector for B"}),
                Lexicon.STRING: ("STRING", {"default": "", "dynamicPrompts": False, "multiline": True}),
            }
        })
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls) -> float:
        if cls.UPDATE:
            return float("nan")
        return super().IS_CHANGED()

    def run(self, **kw) -> Tuple[bool]:
        raw = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        r_x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, None)
        r_y = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, None)
        r_z = parse_param(kw, Lexicon.Z, EnumConvertType.FLOAT, None)
        r_w = parse_param(kw, Lexicon.W, EnumConvertType.FLOAT, None)
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumConvertType.BOOLEAN.name)
        xyzw = parse_param(kw, Lexicon.IN_A+Lexicon.IN_A, EnumConvertType.VEC4, [(0, 0, 0, 0)])
        seed = parse_param(kw, Lexicon.SEED, EnumConvertType.INT, 0, 0)
        yyzw = parse_param(kw, Lexicon.IN_B+Lexicon.IN_B, EnumConvertType.VEC4, [(1, 1, 1, 1)])
        x_str = parse_param(kw, Lexicon.STRING, EnumConvertType.STRING, "")
        params = list(zip_longest_fill(raw, r_x, r_y, r_z, r_w, typ, xyzw, seed, yyzw, x_str))
        results = []
        pbar = ProgressBar(len(params))
        for idx, (raw, r_x, r_y, r_z, r_w, typ, xyzw, seed, yyzw, x_str) in enumerate(params):
            typ = EnumConvertType[typ]
            default = [x_str]
            default2 = None
            if typ not in [EnumConvertType.STRING, EnumConvertType.LIST, \
                        EnumConvertType.DICT,\
                        EnumConvertType.IMAGE, EnumConvertType.LATENT, \
                        EnumConvertType.ANY, EnumConvertType.MASK]:
                a, b, c, d = xyzw
                a2, b2, c2, d2 = yyzw
                default = (a if r_x is None else r_x,
                    b if r_y is None else r_y,
                    c if r_z is None else r_z,
                    d if r_w is None else r_w)
                default2 = (a2, b2, c2, d2)

            val = parse_value(raw, typ, default)
            val2 = parse_value(default2, typ, default2)
            typ = EnumConvertType.VEC4 if typ in [EnumConvertType.VEC4, EnumConvertType.VEC3, \
                                                EnumConvertType.VEC2, EnumConvertType.FLOAT] \
                                                else EnumConvertType.VEC4INT

            # check if set to randomize....
            self.UPDATE = False
            if seed != 0 and isinstance(val, (tuple, list,)) and isinstance(val2, (tuple, list,)):
                self.UPDATE = True
                # val = list(val) if isinstance(val, (tuple, list,)) else [val]
                # val2 = list(val2) if isinstance(val2, (tuple, list,)) else [val2]
                for i in range(len(val)):
                    mx = max(val[i], val2[i])
                    mn = min(val[i], val2[i])
                    if mn == mx:
                        val[i] = mn
                    else:
                        random.seed(seed)
                        if typ == EnumConvertType.VEC4:
                            val[i] = mn + random.random() * (mx - mn)
                        else:
                            logger.debug(f"{i}, {mx}, {mn}")
                            val[i] = random.randint(mn, mx)

            extra = parse_value(val, typ, val)
            ret = [val]
            ret.extend(extra)
            results.append(ret)
            pbar.update_absolute(idx)
        return [x for x in zip(*results)]

class WaveGeneratorNode(JOVBaseNode):
    NAME = "WAVE GEN (JOV) 🌊"
    NAME_PRETTY = "WAVE GEN (JOV) 🌊"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = (Lexicon.FLOAT, Lexicon.INT, )
    SORT = 90
    DESCRIPTION = """
The `Wave Generator` node produces waveforms like sine, square, or sawtooth with adjustable frequency, amplitude, phase, and offset. It's handy for creating oscillating patterns or controlling animation dynamics. This node emits both continuous floating-point values and integer representations of the generated waves.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.WAVE: (EnumWave._member_names_, {"default": EnumWave.SIN.name}),
                Lexicon.FREQ: ("FLOAT", {"default": 1, "min": 0, "step": 0.01, "max": 10000000000000000}),
                Lexicon.AMP: ("FLOAT", {"default": 1, "min": 0, "step": 0.01, "max": 10000000000000000}),
                Lexicon.PHASE: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001, "max": 1.0}),
                Lexicon.OFFSET: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001, "max": 1.0}),
                Lexicon.TIME: ("FLOAT", {"default": 0, "min": 0, "step": 0.000001}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[float, int]:
        op = parse_param(kw, Lexicon.WAVE, EnumConvertType.STRING, EnumWave.SIN.name)
        freq = parse_param(kw, Lexicon.FREQ, EnumConvertType.FLOAT, 1, 0.0001)
        amp = parse_param(kw, Lexicon.AMP, EnumConvertType.FLOAT, 1, 0.0001)
        phase = parse_param(kw, Lexicon.PHASE, EnumConvertType.FLOAT, 0)
        shift = parse_param(kw, Lexicon.OFFSET, EnumConvertType.FLOAT, 0)
        delta_time = parse_param(kw, Lexicon.TIME, EnumConvertType.FLOAT, 0, 0)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        abs = parse_param(kw, Lexicon.ABSOLUTE, EnumConvertType.BOOLEAN, False)
        results = []
        params = list(zip_longest_fill(op, freq, amp, phase, shift, delta_time, invert, abs))
        pbar = ProgressBar(len(params))
        for idx, (op, freq, amp, phase, shift, delta_time, invert, abs) in enumerate(params):
            # freq = 1. / freq
            if invert:
                amp = -amp
            val = wave_op(op, phase, freq, amp, shift, delta_time)
            if abs:
                val = np.abs(val)
            results.append([val, int(val)])
            pbar.update_absolute(idx)
        return *list(zip(*results)),
        # return [list(x) for x in (zip(*results))]

'''
class ParameterNode(JOVBaseNode):
    NAME = "PARAMETER (JOV) ⚙️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    SORT = 100
    DESCRIPTION = """

"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PASS_IN: (JOV_TYPE_ANY, {"default": None}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, ident, **kw) -> Tuple[Any]:
        return kw[Lexicon.PASS_IN],
'''