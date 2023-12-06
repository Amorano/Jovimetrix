"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import gc
import json
from typing import Any

import torch

from Jovimetrix.jovimetrix import Logger, JOVBaseNode, WILDCARD
from Jovimetrix.sup import stream

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
                "host": ("STRING", {"default": ""}),
                "port": ("INT", {"default": 7227}),
            }}

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def run(self, log: str, host: str, port: int, **kw) -> Any:
        if log == "ERROR":
            Logger._LEVEL = 0
        elif log == "WARN":
            Logger._LEVEL = 1
        elif log == "INFO":
            Logger._LEVEL = 2
        elif log == "DEBUG":
            Logger._LEVEL = 3

        stream.STREAMPORT = port
        stream.STREAMHOST = host

        o = kw.get('o', None)
        return (o, )

class DisplayDataNode(JOVBaseNode):
    """Display any data."""

    NAME = "📊 Display Data (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Display any data"
    SORT = 100
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
                    "source": (WILDCARD, {}),
                },
                "optional": {

                }}

    def run(self, source=None) -> dict:
        value = 'None'
        if source is not None:
            try:
                value = json.dumps(source, indent=2, sort_keys=True)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = 'source could not be serialized.'

        return {"ui": {"text": (value,)}}

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
