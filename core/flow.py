"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Logic and Code flow nodes
"""

import os
from enum import Enum
from typing import Any

from loguru import logger

from comfy.utils import ProgressBar
from nodes import interrupt_processing

from Jovimetrix import comfy_message, \
    ComfyAPIMessage, JOVBaseNode, TimedOutException, JOV_WEB_RES_ROOT, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import zip_longest_fill
from Jovimetrix.core.calc import EnumConvertType, parse_parameter

# =============================================================================

JOV_CATEGORY = "FLOW"

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

# =============================================================================

class DelayNode(JOVBaseNode):
    NAME = "DELAY (JOV) ✋🏽"
    NAME_URL = "DELAY ✋🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"[{NAME_URL}]({JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md)"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.ROUTE,)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PASS_IN: (WILDCARD, {"default": None}),
            Lexicon.TIMER: ("INT", {"step": 1, "default" : 0, "min": -1}),
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, ident, **kw) -> tuple[Any]:
        delay = parse_parameter(Lexicon.TIMER, kw, 0, EnumConvertType.INT, -1, JOV_DELAY_MAX)[0]
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
        return (kw[Lexicon.PASS_IN], )

"""
class HoldValueNode(JOVBaseNode):
    NAME = "HOLD VALUE (JOV) 🫴🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-hold"
    DESC = "When engaged will send the last value it had even with new values arriving."
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.ROUTE,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PASS_IN: (WILDCARD, {"default": None}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__last_value = None

    def run(self, **kw) -> tuple[Any]:
        obj = parse_parameter(Lexicon.PASS_IN, kw, None, EnumConvertType.ANY)
        hold = parse_parameter(Lexicon.WAIT, kw, False, EnumConvertType.BOOLEAN)
        params = [tuple(x) for x in zip_longest_fill(obj, hold)]
        pbar = ProgressBar(len(params))
        results = []
        for idx, (obj, hold) in enumerate(params):
            if self.__last_value is None or hold:
                self.__last_value = o
            vals.append(val)
            results.append(good if all(val) else fail)
            pbar.update_absolute(idx)
        return results,


        return (self.__last_value,)
"""

class ComparisonNode(JOVBaseNode):
    NAME = "COMPARISON (JOV) 🕵🏽"
    NAME_URL = "COMPARISON 🕵🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"[{NAME_URL}]({JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md)"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    # DESC = "Compare two inputs: A=B, A!=B, A>B, A>=B, A<B, A<=B"
    RETURN_TYPES = (WILDCARD, WILDCARD,)
    RETURN_NAMES = (Lexicon.ANY, Lexicon.VEC, )
    OUTPUT_IS_LIST = (True, True, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None, "tooltip":"Master Comparator"}),
            Lexicon.IN_B: (WILDCARD, {"default": None, "tooltip":"Secondary Comparator"}),
            Lexicon.COMP_A: (WILDCARD, {"default": None}),
            Lexicon.COMP_B: (WILDCARD, {"default": None}),
            Lexicon.COMPARE: (EnumComparison._member_names_, {"default": EnumComparison.EQUAL.name}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[bool]:
        A = parse_parameter(Lexicon.IN_A, kw, None, EnumConvertType.ANY)
        B = parse_parameter(Lexicon.IN_B, kw, None, EnumConvertType.ANY)
        good = parse_parameter(Lexicon.COMP_A, kw, None, EnumConvertType.ANY)
        fail = parse_parameter(Lexicon.COMP_B, kw, None, EnumConvertType.ANY)
        flip = parse_parameter(Lexicon.FLIP, kw, False, EnumConvertType.BOOLEAN)
        op = parse_parameter(Lexicon.COMPARE, kw, EnumComparison.EQUAL.name, EnumConvertType.STRING)
        params = [tuple(x) for x in zip_longest_fill(A, B, op, flip)]
        pbar = ProgressBar(len(params))
        vals = []
        results = []
        for idx, (A, B, op, flip) in enumerate(params):
            if not isinstance(A, (list, set, tuple)):
                A = [A]
            if not isinstance(B, (list, set, tuple)):
                B = [B]
            size = min(4, max(len(A), len(B))) - 1
            typ = [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4][size]
            val_a = parse_parameter(typ, A, [A[-1]] * size)
            val_b = parse_parameter(typ, B, [B[-1]] * size)
            if flip:
                val_a, val_b = val_b, val_a

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
            vals.append(val)
            results.append(good if all(val) else fail)
            pbar.update_absolute(idx)
        return results, vals,
