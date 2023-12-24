"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

from typing import Any

import torch

from Jovimetrix import deep_merge_dict, \
    JOVBaseNode, Logger, Lexicon, \
    IT_REQUIRED, WILDCARD

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

class AkashicNode(JOVBaseNode):
    NAME = "AKASHIC (JOV) 📓"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Display the top level attributes of an output"
    INPUT_IS_LIST = True
    RETURN_TYPES = ('AKASHIC',)
    RETURN_NAMES = (Lexicon.DATA,)
    RETURN_TYPES = (WILDCARD, 'AKASHIC', )
    RETURN_NAMES = (Lexicon.PASS_OUT, Lexicon.IO)
    OUTPUT_IS_LIST = (True, False )
    SORT = 50

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PASS_IN: (WILDCARD, {})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def __parse(self, s) -> dict[str, Any]:
        def handle_dict(d) -> dict[str, Any]:
            result = {"t": "dict", "items": {}}
            for key, value in d.items():
                result["items"][key] = self.__parse(value)
            return result

        def handle_list(cls) -> dict[str, Any]:
            result = {"t": repr(type(cls)), "items": []}
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
            return {"t": meh, "value": s}

    def run(self, **kw) -> tuple[Any, Any]:
        o = kw.get(Lexicon.PASS_IN, None)
        if o is None:
            return (o, {})

        value = self.__parse(o)
        if kw.get(Lexicon.OUTPUT, False):
            Logger.dump(value)
        return (o, value,)

class ValueGraphNode(JOVBaseNode):
    NAME = "VALUE GRAPH (JOV) 📈"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Graphs historical execution run values"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    SORT = 100

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PASS_IN: (WILDCARD, {})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def __parse(self, s) -> dict[str, Any]:
        def handle_dict(d) -> dict[str, Any]:
            result = {"t": "dict", "items": {}}
            for key, value in d.items():
                result["items"][key] = self.__parse(value)
            return result

        def handle_list(cls) -> dict[str, Any]:
            result = {"t": repr(type(cls)), "items": []}
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
            return {"t": meh, "value": s}

    def run(self, **kw) -> tuple[Any, Any]:
        o = kw.get(Lexicon.PASS_IN, None)
        if o is None:
            return (o, {})

        value = self.__parse(o)
        if kw.get(Lexicon.OUTPUT, False):
            Logger.dump(value)
        return (o, value,)
