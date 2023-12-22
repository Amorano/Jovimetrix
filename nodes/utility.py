"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

from typing import Any

import torch

from Jovimetrix import deep_merge_dict, \
    JOVBaseNode, Logger, Lexicon, \
    IT_REQUIRED, IT_PASS_IN, WILDCARD

# =============================================================================

class OptionsNode(JOVBaseNode):
    NAME = "OPTIONS (JOV) ⚙️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Change Jovimetrix Global Options"
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.PASS_OUT, )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "optional": {
                Lexicon.PASS_IN: (WILDCARD, {"default": None}),
                Lexicon.LOG: (["ERROR", "WARN", "INFO", "DEBUG", "SPAM"], {"default": "ERROR"}),
                #"host": ("STRING", {"default": ""}),
                #"port": ("INT", {"min": 0, "step": 1, "default": 7227}),
            }}
        return deep_merge_dict(IT_REQUIRED, d)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def run(self, **kw) -> tuple[Any]:
        log = kw.get(Lexicon.LOG, 0)

        if log == "ERROR":
            Logger._LEVEL = 0
        elif log == "WARN":
            Logger._LEVEL = 1
        elif log == "INFO":
            Logger._LEVEL = 2
        elif log == "DEBUG":
            Logger._LEVEL = 3
        elif log == "SPAM":
            Logger._LEVEL = 4

        #stream.STREAMPORT = port
        #stream.STREAMHOST = host

        o = kw.get(Lexicon.PASS_IN, None)
        return (o, )

class DebugNode(JOVBaseNode):
    """Display any data."""

    NAME = "DEBUG (JOV) 🪲"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Debug data"
    RETURN_TYPES = (WILDCARD, WILDCARD, )
    RETURN_NAMES = (Lexicon.PASS_OUT, Lexicon.IO)
    OUTPUT_NODE = True
    SORT = 50

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PASS_IN: (WILDCARD, {}),
            Lexicon.OUTPUT : ("BOOLEAN", {"default": True}),
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def __parse(self, s) -> dict[str, Any]:
        def handle_dict(d) -> dict[str, Any]:
            result = {"type": "dict", "items": {}}
            for key, value in d.items():
                result["items"][key] = self.__parse(value)
            return result

        def handle_list(cls) -> dict[str, Any]:
            result = {"type": repr(type(cls)), "items": []}
            for item in s:
                result["items"].append(self.__parse(item))
            return result

        if isinstance(s, dict):
            return handle_dict(s)
        elif isinstance(s, (tuple, set, list,)):
            return handle_list(s)
        elif isinstance(s, torch.Tensor):
            return {"Tensor": f"{s.shape}"}
        else:
            meh = ''.join(repr(type(s)).split("'")[1:2])
            return {"type": meh, "value": s}

    def run(self, **kw) -> tuple[Any, Any]:
        o = kw.get(Lexicon.PASS_IN, None)
        if o is None:
            return (o, {})

        value = self.__parse(o)
        if kw.get(Lexicon.OUTPUT, False):
            Logger.dump(value)
        return (o, value,)

class AkashicNode(JOVBaseNode):
    NAME = "AKASHIC (JOV) 📓"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Display the top level attributes of an input"
    RETURN_TYPES = ('AKASHIC',)
    RETURN_NAMES = (Lexicon.DATA,)
    OUTPUT_IS_LIST = (True, )
    OUTPUT_NODE = True
    SORT = 100

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PASS_IN)

    def run(self, **kw) -> tuple[dict]:
        data = {}
        return (data, )
