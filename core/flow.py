"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Logic and Code flow nodes
"""

import os
from enum import Enum
from typing import Any, Tuple

from loguru import logger

from comfy.utils import ProgressBar
from nodes import interrupt_processing

from Jovimetrix import comfy_message, \
    ComfyAPIMessage, JOVBaseNode, TimedOutException, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_param, parse_value, zip_longest_fill
from Jovimetrix.core.calc import EnumConvertType

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
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.PASS_OUT,)
    DESCRIPTION = """
Delay node used to introduce pauses in the workflow. It accepts an optional input to pass through and a timer parameter to specify the duration of the delay. If no timer is provided, it defaults to a maximum delay. During the delay, it periodically checks for messages to interrupt the delay. Once the delay is completed, it returns the input passed to it.
"""

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

class ComparisonNode(JOVBaseNode):
    NAME = "COMPARISON (JOV) 🕵🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (WILDCARD, WILDCARD,)
    RETURN_NAMES = (Lexicon.ANY, Lexicon.VEC,)
    DESCRIPTION = """
The Comparison node evaluates two inputs based on a specified operation. It accepts two inputs (A and B), comparison operators, and optional values for successful and failed comparisons. The node performs the specified operation element-wise between corresponding elements of A and B. If the comparison is successful for all elements, it returns the success value; otherwise, it returns the failure value. The node supports various comparison operators such as EQUAL, GREATER_THAN, LESS_THAN, AND, OR, IS, IN, etc.
"""

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
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[Any, Any]:
        A = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, None)
        B = parse_param(kw, Lexicon.IN_B, EnumConvertType.ANY, None)
        good = parse_param(kw, Lexicon.COMP_A, EnumConvertType.ANY, None)
        fail = parse_param(kw, Lexicon.COMP_B, EnumConvertType.ANY, None)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        op = parse_param(kw, Lexicon.COMPARE, EnumConvertType.STRING, EnumComparison.EQUAL.name)
        params = list(zip_longest_fill(A, B, op, flip))
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
            vals.append(val)
            results.append(good if all([bool(v) for v in val]) else fail)
            pbar.update_absolute(idx)
        return results, vals,
