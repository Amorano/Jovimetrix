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

import comfy
from server import PromptServer
import nodes

from Jovimetrix import ComfyAPIMessage, JOVBaseNode, TimedOutException, \
    JOV_HELP_URL, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import zip_longest_fill, convert_parameter

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/FLOW"

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

class DelayNode(JOVBaseNode):
    NAME = "DELAY (JOV) ✋🏽"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Delay traffic. Electrons on the data bus go round."
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
            "id": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/FLOW#-delay")

    @staticmethod
    def parse_q(id, delay: int, forced:bool=False)-> bool:
        pbar = comfy.utils.ProgressBar(delay)
        # if longer than X seconds, pop up the "cancel continue"
        if delay > JOV_DELAY_MIN or forced:
            PromptServer.instance.send_sync("jovi-delay-user", {"id": id, "timeout": delay})

        step = 0
        while (step := step + 1) <= delay:
            try:
                if delay > JOV_DELAY_MIN or forced:
                    data = ComfyAPIMessage.poll(id, timeout=1)
                    if data.get('cancel', None):
                        nodes.interrupt_processing(True)
                        logger.warning(f"render cancelled delay: {id}")
                    else:
                        logger.info(f"render continued delay: {id}")
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

    def run(self, id, **kw) -> tuple[Any]:
        o = kw.get(Lexicon.PASS_IN, [None])
        wait = kw.get(Lexicon.WAIT, [False])[0]
        delay = min(kw.get(Lexicon.TIMER, 0), [JOVI_DELAY_MAX])[0]

        if wait:
            cancel = False
            while not cancel:
                loop_delay = delay
                if loop_delay == 0:
                    loop_delay = JOVI_DELAY_MAX
                cancel = DelayNode.parse_q(id, loop_delay, True)
            return o

        if delay != self.__delay:
            self.__delay = delay
            self.__delay = max(0, min(self.__delay, JOVI_DELAY_MAX))

        loops = int(self.__delay)
        if (remainder := self.__delay - loops) > 0:
            time.sleep(remainder)

        if loops > 0:
            cancel = DelayNode.parse_q(id, loops)
        return o

class HoldValueNode(JOVBaseNode):
    NAME = "HOLD VALUE (JOV) 🫴🏽"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "When engaged will send the last value it had even with new values arriving."
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
        return Lexicon._parse(d, JOV_HELP_URL + "/FLOW#-hold")

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
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Compare two inputs: A=B, A!=B, A>B, A>=B, A<B, A<=B"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = (Lexicon.UNKNOWN, )
    OUTPUT_IS_LIST = (True, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.COMPARE: (EnumComparison._member_names_, {"default": EnumComparison.EQUAL.name}),
            Lexicon.IN_B: (WILDCARD, {"default": None}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/FLOW#-comparison")

    def run(self, **kw) -> tuple[bool]:
        result = []
        A = kw.get(Lexicon.IN_A, [None])
        B = kw.get(Lexicon.IN_B, [None])
        flip = kw.get(Lexicon.FLIP, [None])
        op = kw.get(Lexicon.COMPARE, [None])
        params = [tuple(x) for x in zip_longest_fill(A, B, op, flip)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (a, b, op, flip) in enumerate(params):
            if type(a) == tuple and type(b) == tuple:
                if (short := len(a) - len(b)) > 0:
                    b = list(b) + [0] * short
            typ_a, val_a = convert_parameter(a)
            _, val_b = convert_parameter(b)
            if flip:
                a, b = b, a
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

            val = [typ_a[i](v) for i, v in enumerate(val)]
            if len(val) == 1:
                result.append(val[0])
            else:
                result.append(tuple(val))
            # logger.debug("{} {}", result, val)
            pbar.update_absolute(idx)

        return (result, )

class SelectNode(JOVBaseNode):
    NAME = "SELECT (JOV) 🤏🏽"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Select an item from a user explicit list of inputs."
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
            "id": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/FLOW#-select")

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__index = 0

    def run(self, id, **kw) -> None:
        reset = kw.get(Lexicon.RESET, False)
        try:
            data = ComfyAPIMessage.poll(id, timeout=0)
            if (cmd := data.get('cmd', None)) is not None:
                if cmd == 'reset':
                    reset = True
        except TimedOutException as e:
            pass
        except Exception as e:
            logger.error(str(e))

        if reset:
            self.__index = 0

        count = 0
        vals = []
        while 1:
            who = f"{Lexicon.UNKNOWN}_{count+1}"
            if (val := kw.get(who, None)) is None:
                break
            vals.append(val)
            count += 1

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
