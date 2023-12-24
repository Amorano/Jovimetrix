"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Logic and Code flow nodes
"""

import time

from enum import Enum
from typing import Any

from Jovimetrix import deep_merge_dict, \
    Logger, JOVBaseNode, Lexicon, \
    JOV_MAX_DELAY, IT_REQUIRED, IT_AB, WILDCARD

# =============================================================================

class EnumComparison(Enum):
    EQUAL = 0
    NOT_EQUAL = 1
    LESS_THAN = 2
    LESS_THAN_OR_EQUAL = 3
    GREATER_THAN = 4
    GREATER_THAN_OR_EQUAL = 5

class DelayNode(JOVBaseNode):
    NAME = "DELAY (JOV) ✋🏽"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/FLOW"
    DESCRIPTION = "Delay traffic. Electrons on the data bus go round."
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.ROUTE,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PASS_IN: (WILDCARD, {"default": None}),
            Lexicon.WAIT: ("FLOAT", {"step": 0.01, "default" : 0}),
            Lexicon.RESET: ("BOOLEAN", {"default": False})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def __init__(self) -> None:
        self.__delay = 0

    def run(self, **kw) -> tuple[Any]:
        o = kw.get(Lexicon.PASS_IN, None)
        delay = kw.get(Lexicon.DELAY, 0)
        reset = kw.get(Lexicon.RESET, False)

        if reset:
            self.__delay = 0
            return (self, )

        if delay != self.__delay:
            self.__delay = delay
            self.__delay = max(0, min(self.__delay, JOV_MAX_DELAY))

        time.sleep(self.__delay)
        return (o,)

class ComparisonNode(JOVBaseNode):
    """Compare two inputs."""

    NAME = "COMPARISON (JOV) 🕵🏽"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/FLOW"
    DESCRIPTION = "Compare two inputs"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = (Lexicon.BOOLEAN, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.COMPARE: (EnumComparison._member_names_, {"default": EnumComparison.EQUAL.name}),
            Lexicon.IN_B: (WILDCARD, {"default": None})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        compare = kw.get(Lexicon.COMPARE, EnumComparison.A_EQUALS_B)
        A = kw.get(Lexicon.IN_A, None)
        B = kw.get(Lexicon.IN_B, None)

        match compare:
            case EnumComparison.A_EQUALS_B:
                return (A == B,)
            case EnumComparison.A_GREATER_THAN_B:
                return (A > B,)
            case EnumComparison.A_GREATER_THAN_OR_EQUAL_TO_B:
                return (A >= B,)
            case EnumComparison.A_LESS_THAN_B:
                return (A < B,)
            case EnumComparison.A_LESS_THAN_OR_EQUAL_TO_B:
                return (A <= B,)
            case EnumComparison.A_NOT_EQUAL_TO_B:
                return (A != B,)
        return (False,)

class IfThenElseNode(JOVBaseNode):
    NAME = "IF-THEN-ELSE (JOV) 🔀"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/FLOW"
    DESCRIPTION = "IF <valid> then A else B"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = (Lexicon.RESULT, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.TRUE: (WILDCARD, {"default": None}),
            Lexicon.FALSE: (WILDCARD, {"default": None}),
            Lexicon.CONDITION: ("BOOLEAN", {"default": False}),
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        o = kw.get(Lexicon.CONDITION, False)
        T = kw.get(Lexicon.TRUE, None)
        F = kw.get(Lexicon.FALSE, None)
        if T is None or F is None:
            return (None,)
        return (T if o else F,)
