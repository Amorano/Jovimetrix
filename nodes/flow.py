"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Logic and Code flow nodes
"""

import time

from enum import Enum
from typing import Any

from Jovimetrix import deep_merge_dict, \
    Logger, JOVBaseNode, \
    JOV_MAX_DELAY, IT_REQUIRED, WILDCARD

# =============================================================================

class EnumComparison(Enum):
    A_EQUALS_B = 0
    A_NOT_EQUAL_TO_B = 1
    A_LESS_THAN_B = 2
    A_LESS_THAN_OR_EQUAL_TO_B = 3
    A_GREATER_THAN_B = 4
    A_GREATER_THAN_OR_EQUAL_TO_B = 5

class EnumLogicGate(Enum):
    A_AND_B = 6
    A_NAND_B = 7
    A_OR_B = 8
    A_NOR_B = 9
    A_XOR_B = 10
    A_XNOR_B = 11
    A_NOT_B = 12

class RouteNode(JOVBaseNode):
    NAME = "ROUTE (JOV) 🚌"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/FLOW"
    DESCRIPTION = "Pass-thru, delay, or hold traffic. Electrons on the data bus go round."
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🚌",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            "o": (WILDCARD, {"default": None}),
            "delay": ("FLOAT", {"step": 0.01, "default" : 0}),
            "hold": ("BOOLEAN", {"default": False}),
            "reset": ("BOOLEAN", {"default": False})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def __init__(self) -> None:
        self.__delay = 0

    def run(self, o: Any, delay: float, hold: bool, reset: bool) -> Any:
        ''' @TODO
        t = threading.Thread(target=self.__run, daemon=True)
        t.start()
        '''
        if reset:
            self.__delay = 0
            return (self, )

        if hold:
            return(None,)

        if delay != self.__delay:
            self.__delay = delay
            self.__delay = max(0, min(self.__delay, JOV_MAX_DELAY))

        time.sleep(self.__delay)
        return (o,)

    def __run(self) -> None:
        while self.__hold:
            time.sleep(0.1)

class ComparisonNode(JOVBaseNode):
    """Compare two inputs."""

    NAME = "COMPARISON (JOV) 🕵🏽"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/FLOW"
    DESCRIPTION = "Compare two inputs"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("🅱️")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "A": (WILDCARD, {"default": None}),
                "B": (WILDCARD, {"default": None}),
                "comparison": (EnumComparison._member_names_, {"default": EnumComparison.A_EQUALS_B.value}),
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, A: Any, B: Any, comparison: EnumComparison) -> tuple[bool]:
        match comparison:
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
    RETURN_NAMES = "❔"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "o": ("BOOLEAN", {"default": False}),
                "🇹": (WILDCARD, {"default": None}),
                "🇫": (WILDCARD, {"default": None}),
            },
        }

    def run(self, o:bool, **kw) -> tuple[bool]:
        T = kw.get('🇹', None)
        F = kw.get('🇫', None)
        if T is None or F is None:
            return (None,)
        return (T if o else F,)
