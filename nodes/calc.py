"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Calculation
"""

import math
from enum import Enum
from typing import Any
from collections import Counter

from scipy.special import gamma

from Jovimetrix import zip_longest_fill, deep_merge_dict, \
    JOVBaseNode, Logger, Lexicon, \
    IT_REQUIRED, IT_AB, WILDCARD

# =============================================================================

class EnumConvertType(Enum):
    BOOLEAN = 0
    INTEGER = 1
    FLOAT   = 2
    VEC2 = 3
    VEC3 = 4
    VEC4 = 5
    STRING = 6
    TUPLE = 7

class ConversionNode(JOVBaseNode):
    """Convert A to B."""

    NAME = "CONVERT (JOV) 🧬"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CALC"
    DESCRIPTION = "Convert A to B."
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN, )
    SORT = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            # Lexicon.TYPE: (EnumConvertType._member_names_, {"default": EnumConvertType.INT.name})
        }}
        return deep_merge_dict(IT_REQUIRED, IT_AB, d)

    def run(self, **kw) -> tuple[bool]:
        A = kw.get(Lexicon.IN_A, None)
        typ = kw.get("JTYPE", None)

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
    EXP = 20
    # COMPOUND
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
    EnumUnaryOperation.COS_H: lambda x: math.cosh(x),
    EnumUnaryOperation.SIN_H: lambda x: math.sinh(x),
    EnumUnaryOperation.TAN_H: lambda x: math.tanh(x),
    EnumUnaryOperation.RADIANS: lambda x: math.radians(x),
    EnumUnaryOperation.DEGREES: lambda x: math.degrees(x),
    EnumUnaryOperation.GAMMA: lambda x: gamma(x)
}

class CalcUnaryOPNode(JOVBaseNode):
    """Perform a Unary Operation on an input."""

    NAME = "CALC OP UNARY (JOV) 🎲"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CALC"
    DESCRIPTION = "Perform a Unary Operation on an input"
    INPUT_IS_LIST = True
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN, )
    OUTPUT_IS_LIST = (True, )
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.TYPE: (EnumUnaryOperation._member_names_, {"default": EnumUnaryOperation.ABS.name})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        result = []
        data = kw.get(Lexicon.IN_A, [0])
        op = kw.get(Lexicon.TYPE, [EnumUnaryOperation.ABS.value])
        # print(data, op)
        for data, op in zip_longest_fill(data, op):

            if data is None:
                result.append((0, ))
                continue

            if not isinstance(data, (list, tuple, set,)):
                data = [data]

            typ = []
            val = []
            for v in data:
                # print(v, type(v), type(v) == float, type(v) == int)
                t = type(v)
                if t == int:
                    t = float
                try:
                    v = float(v)
                except Exception as e:
                    Logger.debug(str(e))
                    v = 0
                typ.append(t)
                val.append(v)

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
                        val = [math.sqrt(sum(val))]
                case EnumUnaryOperation.NORMALIZE:
                    if len(val) == 1:
                        val = [val]
                    else:
                        m = max(val)
                        val = [v / m for v in val]
                case _:
                    # Apply unary operation to each item in the list
                    ret = []
                    for v in val:
                        try:
                            v = OP_UNARY[op](v)
                        except Exception as e:
                            Logger.spam(str(e))
                            v = 0
                        ret.append(v)
                    val = ret

            # Reconvert the values back to their original types
            val = [typ[i](v) for i, v in enumerate(val)]
            if len(val) == 1:
                result.append(val[0])
            else:
                result.append(tuple(val))
            print(result, val)
        return (result, )

class EnumBinaryOperation(Enum):
    ADD = 0
    SUBTRACT = 1
    MULTIPLY   = 2
    DIVIDE = 3
    DIVIDE_FLOOR = 4
    MODULUS = 5
    POWER = 6
    # LOGIC
    # NOT = 10
    AND = 11
    NAND = 12
    OR = 13
    NOR = 14
    XOR = 15
    XNOR = 16
    # BITS
    # BIT_NOT = 20
    BIT_AND = 21
    BIT_NAND = 22
    BIT_OR = 23
    BIT_NOR = 24
    BIT_XOR = 25
    BIT_XNOR = 26
    BIT_LSHIFT = 27
    BIT_RSHIFT = 28

class CalcBinaryOPNode(JOVBaseNode):
    """Perform a Binary Operation on two inputs."""

    NAME = "CALC OP BINARY (JOV) 🌟"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CALC"
    DESCRIPTION = "Perform a Binary Operation on two inputs"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.UNKNOWN, )
    SORT = 20

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.IN_B: (WILDCARD, {"default": None}),
            Lexicon.TYPE: (EnumBinaryOperation._member_names_, {"default": EnumBinaryOperation.ADD.name})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        A = kw.get(Lexicon.IN_A, None)
        typ = kw.get("JTYPE", None)

        return (None, )
