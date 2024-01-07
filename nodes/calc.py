"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Calculation
"""

import math
from enum import Enum
from collections import Counter

from scipy.special import gamma
from loguru import logger

from Jovimetrix import JOVBaseNode, IT_REQUIRED, WILDCARD, IT_FLIP
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import zip_longest_fill, convert_parameter, deep_merge_dict

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
    # RETURN_NAMES = (Lexicon.UNKNOWN, )
    SORT = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.TYPE: (["STRING", "BOOLEAN", "INT", "FLOAT", "VEC2", "VEC3", "VEC4"], {"default": "BOOLEAN"})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        typ = kw.pop(Lexicon.TYPE)
        a = next(iter(kw.values()))
        size = len(a) if type(a) == tuple else 0
        # logger.debug("{} {}", size, a)
        if typ in ["STRING", "FLOAT"]:
            if size > 0:
                return ((a[0]), )
            return ((a), )
        elif typ == "BOOLEAN":
            if size > 0:
                return (bool(a[0]), )
            return (bool(a), )
        elif typ == "INT":
            if size > 0:
                return (int(a[0]), )
            return (int(a), )

        if typ == "VEC2":
            if size > 1:
                return ((a[0], a[1]), )
            elif size > 0:
                return ((a[0], a[0]), )
            return ((a, a), )

        if typ == "VEC3":
            if size > 2:
                return ((a[0], a[1], a[2]), )
            elif size > 1:
                return ((a[0], a[1], a[1]), )
            elif size > 0:
                return ((a[0], a[0], a[0]), )
            return ((a, a, a), )

        if typ == "VEC4":
            if size > 3:
                return ((a[0], a[1], a[2], a[3]), )
            elif size > 2:
                return ((a[0], a[1], a[2], a[2]), )
            elif size > 1:
                return ((a[0], a[1], a[1], a[1]), )
            elif size > 0:
                return ((a[0], a[0], a[0], a[0]), )
            return ((a, a, a, a), )

        return (("nan"), )

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
            Lexicon.FUNC: (EnumUnaryOperation._member_names_, {"default": EnumUnaryOperation.ABS.name})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        result = []
        data = kw.get(Lexicon.IN_A, [0])
        op = kw.get(Lexicon.FUNC, [EnumUnaryOperation.ABS])
        for data, op in zip_longest_fill(data, op):
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
        return (result, )

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

class CalcBinaryOPNode(JOVBaseNode):
    """Perform a Binary Operation on two inputs."""

    NAME = "CALC OP BINARY (JOV) 🌟"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CALC"
    DESCRIPTION = "Perform a Binary Operation on two inputs"
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
        return deep_merge_dict(IT_REQUIRED, d, IT_FLIP)

    def run(self, **kw) -> tuple[bool]:
        result = []
        A = kw[Lexicon.IN_A]
        B = kw[Lexicon.IN_B]
        flip = kw[Lexicon.FLIP]
        op = kw[Lexicon.FUNC]
        for a, b, op, flip in zip_longest_fill(A, B, op, flip):
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
        return (result, )

class ValueNode(JOVBaseNode):
    """Create a value for most types."""

    NAME = "VALUE (JOV) #️⃣"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CALC"
    DESCRIPTION = "Create a value for most types; also universal constants."
    INPUT_IS_LIST = False
    RETURN_TYPES = (WILDCARD, )
    # RETURN_NAMES = ()
    OUTPUT_IS_LIST = (False, )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.TYPE: (["STRING", "BOOLEAN", "INT", "FLOAT", "VEC2", "VEC3", "VEC4"], {"default": "BOOLEAN"}),
                Lexicon.X: ("FLOAT", {"default": 0}),
                Lexicon.Y: ("FLOAT", {"default": 0}),
                Lexicon.Z: ("FLOAT", {"default": 0}),
                Lexicon.W: ("FLOAT", {"default": 0})
            },
            "hidden": {

            }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        typ = kw[Lexicon.TYPE]
        # should always have an X value
        x = kw[Lexicon.X]
        y = kw.get(Lexicon.Y, 0)
        z = kw.get(Lexicon.Z, 0)
        w = kw.get(Lexicon.W, 0)
        match typ:
            case "VEC2":
                return((x, y,), )
            case "VEC3":
                return((x, y, z, ), )
            case "VEC4":
                return((x, y, z, w, ), )
        return (x, )
