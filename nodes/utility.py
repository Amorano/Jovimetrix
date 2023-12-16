"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import gc
import json
from pickle import TRUE
from typing import Any, Optional

import comfy
import torch

from Jovimetrix import Logger, JOVBaseNode, WILDCARD

# =============================================================================

class RouteNode(JOVBaseNode):
    NAME = "🚌 Route (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Wheels on the BUS pass the data through, around and around."
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🚌",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
            "o": (WILDCARD, {"default": None}),
        }}

    def run(self, o: object) -> object:
        return (o,)

class ClearCacheNode(JOVBaseNode):
    NAME = "🧹 Clear Cache (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Clear the torch cache, and python caches - we need to pay the bills"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🧹",)
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
            "o": (WILDCARD, {}),
        }}

    def run(self, o: Any) -> [object, ]:
        f, t = torch.cuda.mem_get_info()
        Logger.debug(self.NAME, f"total: {t}")
        Logger.debug(self.NAME, "-"* 30)
        Logger.debug(self.NAME, f"free: {f}")

        s = o
        if isinstance(o, dict):
            s = o.copy()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        comfy.model_management.soft_empty_cache()

        f, t = torch.cuda.mem_get_info()
        Logger.debug(self.NAME, f"free: {f}")
        Logger.debug(self.NAME, "-"* 30)
        return (s, )

class OptionsNode(JOVBaseNode):
    NAME = "⚙️ Options (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Change Jovimetrix Global Options"
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = ("🦄", )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required" : {},
            "optional": {
                "o": (WILDCARD, {"default": None}),
                "log": (["ERROR", "WARN", "INFO", "DEBUG"], {"default": "ERROR"}),
                #"host": ("STRING", {"default": ""}),
                #"port": ("INT", {"min": 0, "step": 1, "default": 7227}),
            }}

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def run(self, log: str,  **kw) -> Any:
        if log == "ERROR":
            Logger._LEVEL = 0
        elif log == "WARN":
            Logger._LEVEL = 1
        elif log == "INFO":
            Logger._LEVEL = 2
        elif log == "DEBUG":
            Logger._LEVEL = 3

        #stream.STREAMPORT = port
        #stream.STREAMHOST = host

        o = kw.get('o', None)
        return (o, )

class DebugNode(JOVBaseNode):
    """Display any data."""

    NAME = "🪲 Debug (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Debug data"
    OUTPUT_NODE = True
    SORT = 100
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
                    "source": (WILDCARD, {}),
                },
                "optional": {
                    "console": ("BOOLEAN", {"default": False})
                }}

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    @classmethod
    def parse_value(cls, s) -> dict[str, Any]:
        def handle_dict(d) -> dict[str, Any]:
            result = {"type": "Dictionary", "items": {}}
            for key, value in d.items():
                result["items"][key] = cls.parse_value(value)
            return result

        def handle_list(s) -> dict[str, Any]:
            result = {"type": type(s), "items": []}
            for item in s:
                result["items"].append(cls.parse_value(item))
            return result

        if isinstance(s, dict):
            return handle_dict(s)
        elif isinstance(s, (tuple, set, list,)):
            return handle_list(s)
        elif isinstance(s, torch.Tensor):
            return {"Tensor": f"{s.shape}"}
        else:
            return {"type": repr(type(s)), "value": s}

    def run(self, console:bool, source:list[object]) -> dict:
        # ret = [{"ui": {"text": self.parse_value(s)}} for s in source]
        ret = [self.parse_value(s) for s in source]
        if console:
            value = f'# items: {len(ret)}\n' + json.dumps(ret, indent=2)
            print(value)
        return ret

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
