"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Logic and Code flow nodes
"""

import os
import random
import time
from enum import Enum
from typing import Any

from loguru import logger

from comfy.utils import ProgressBar
from nodes import interrupt_processing

from Jovimetrix import comfy_message, load_help, parse_reset, \
    ComfyAPIMessage, JOVBaseNode, TimedOutException, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_dynamic, zip_longest_fill
from Jovimetrix.core.calc import EnumConvertType, parse_parameter

# =============================================================================

JOV_CATEGORY = "FLOW"

# min amount of time before showing the cancel dialog
JOV_DELAY_MIN = 1
try: JOV_DELAY_MIN = int(os.getenv("JOV_DELAY_MIN", JOV_DELAY_MIN))
except: pass
JOV_DELAY_MIN = max(1, JOV_DELAY_MIN)

# max 10 minutes to start
JOVI_DELAY_MAX = 600
try: JOVI_DELAY_MAX = int(os.getenv("JOVI_DELAY_MAX", JOVI_DELAY_MAX))
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
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-delay"
    DESC = "Delay traffic. Electrons on the data bus go round."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.ROUTE,)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PASS_IN: (WILDCARD, {"default": None}),
            Lexicon.TIMER: ("INT", {"step": 1, "default" : 0}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    @staticmethod
    def parse_q(ident, delay: int, forced:bool=False)-> bool:
        pbar = ProgressBar(delay)
        # if longer than X seconds, pop up the "cancel continue"
        if delay > JOV_DELAY_MIN or forced:
            comfy_message(ident, "jovi-delay-user", {"id": ident, "timeout": delay})

        step = 0
        while (step := step + 1) <= delay:
            try:
                if delay > JOV_DELAY_MIN or forced:
                    data = ComfyAPIMessage.poll(ident, timeout=1)
                    if data.get('cancel', None):
                        interrupt_processing(True)
                        logger.warning(f"render cancelled delay: {ident}")
                    else:
                        logger.info(f"render continued delay: {ident}")
                    return True
                else:
                    time.sleep(1)
                    raise TimedOutException()
            except TimedOutException as e:
                pbar.update_absolute(step)
            except Exception as e:
                logger.error(str(e))
                return True
        return False

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__delay = 0

    def run(self, ident, **kw) -> tuple[Any]:
        o = kw.get(Lexicon.PASS_IN, [None])
        wait = kw.get(Lexicon.WAIT, [False])[0]
        delay = min(kw.get(Lexicon.TIMER, 0), [JOVI_DELAY_MAX])[0]

        if wait:
            cancel = False
            while not cancel:
                loop_delay = delay
                if loop_delay == 0:
                    loop_delay = JOVI_DELAY_MAX
                cancel = DelayNode.parse_q(ident, loop_delay, True)
            return o

        if delay != self.__delay:
            self.__delay = delay
            self.__delay = max(0, min(self.__delay, JOVI_DELAY_MAX))

        loops = int(self.__delay)
        if (remainder := self.__delay - loops) > 0:
            time.sleep(remainder)

        if loops > 0:
            cancel = DelayNode.parse_q(ident, loops)
        return o

class HoldValueNode(JOVBaseNode):
    NAME = "HOLD VALUE (JOV) 🫴🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-hold"
    DESC = "When engaged will send the last value it had even with new values arriving."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
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
        o = kw.get(Lexicon.PASS_IN, [None])
        if self.__last_value is None or not kw.get(Lexicon.WAIT, [False]):
            self.__last_value = o
        return (self.__last_value,)

class ComparisonNode(JOVBaseNode):
    NAME = "COMPARISON (JOV) 🕵🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-comparison"
    DESC = "Compare two inputs: A=B, A!=B, A>B, A>=B, A<B, A<=B"
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
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
        A = kw.get(Lexicon.IN_A, [0])
        B = kw.get(Lexicon.IN_B, [0])
        good = kw.get(Lexicon.COMP_A, [None])
        fail = kw.get(Lexicon.COMP_B, [None])
        flip = kw.get(Lexicon.FLIP, [None])
        op = kw.get(Lexicon.COMPARE, [None])
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

class SelectNode(JOVBaseNode):
    NAME = "SELECT (JOV) 🤏🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-select"
    DESC = "Select an item from a user explicit list of inputs."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    INPUT_IS_LIST = False
    RETURN_TYPES = (WILDCARD, "STRING", "INT", "INT", )
    RETURN_NAMES = (Lexicon.ANY, Lexicon.QUEUE, Lexicon.VALUE, Lexicon.TOTAL, )
    OUTPUT_IS_LIST = (False, False, False, False, )
    SORT = 70

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            #  -1: Random; 0: Sequential; 1..N: explicitly use index
            Lexicon.SELECT: ("INT", {"default": 0, "min": -1, "step": 1}),
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__index = 0

    def run(self, ident, **kw) -> None:
        if parse_reset(ident) > 0 or kw.get(Lexicon.RESET, False):
            self.__index = 0
        vals = parse_dynamic(Lexicon.UNKNOWN, kw)
        count = len(vals)
        select = kw.get(Lexicon.SELECT, 0)
        # clip the index in case it went out of range.
        index = max(0, min(count - 1, self.__index))
        val = None
        if select < 1:
            if select < 0:
                index = int(random.random() * count)
                val = vals[index]
            else:
                val = vals[index]
            index += 1
            if index >= count:
                index = 0
        elif select < count:
            val = vals[index]
        self.__index = index
        return val, vals, self.__index + 1, count,
