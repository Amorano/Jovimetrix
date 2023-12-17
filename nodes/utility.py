"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import gc
from typing import Any, Optional

try:
    import comfy
except:
    pass

import torch

from Jovimetrix import deep_merge_dict, \
    Logger, JOVBaseNode, \
    WILDCARD, IT_REQUIRED

# =============================================================================

class ClearCacheNode(JOVBaseNode):
    NAME = "CACHE (JOV) 🧹"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Clear the torch cache, and python caches - we need to pay the bills"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🧹",)
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            "o": (WILDCARD, {}),
        }}
        return deep_merge_dict(IT_REQUIRED, d)

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
    NAME = "OPTIONS (JOV) ⚙️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Change Jovimetrix Global Options"
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = ("🦄", )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "optional": {
                "o": (WILDCARD, {"default": None}),
                "log": (["ERROR", "WARN", "INFO", "DEBUG", "SPAM"], {"default": "ERROR"}),
                #"host": ("STRING", {"default": ""}),
                #"port": ("INT", {"min": 0, "step": 1, "default": 7227}),
            }}
        return deep_merge_dict(IT_REQUIRED, d)

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
        elif log == "SPAM":
            Logger._LEVEL = 4

        #stream.STREAMPORT = port
        #stream.STREAMHOST = host

        o = kw.get('o', None)
        return (o, )

class DebugNode(JOVBaseNode):
    """Display any data."""

    NAME = "DEBUG (JOV) 🪲"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Debug data"
    RETURN_TYPES = (WILDCARD, WILDCARD, )
    RETURN_NAMES = ("🦄", "💾")
    OUTPUT_NODE = True
    SORT = 100

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            "o": (WILDCARD, {}),
            "dump" : ("BOOLEAN", {"default": True}),
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

    def run(self, o:Optional[object]=None, dump:bool=False) -> object:
        if o is None:
            return (o, {})
        value = self.__parse(o)
        if dump:
            Logger.dump(value)
        return (o, value,)

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
