"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import io
import base64
from typing import Any
import matplotlib.pyplot as plt

import torch
import numpy as np
from PIL import Image
from loguru import logger

from Jovimetrix import JOVBaseNode, IT_REQUIRED, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import deep_merge_dict
from Jovimetrix.sup.image import tensor2pil, pil2tensor

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
            logger._LEVEL = 0
        elif log == "WARN":
            logger._LEVEL = 1
        elif log == "INFO":
            logger._LEVEL = 2
        elif log == "DEBUG":
            logger._LEVEL = 3
        elif log == "SPAM":
            logger._LEVEL = 4

        #stream.STREAMPORT = port
        #stream.STREAMHOST = host

        o = kw.get(Lexicon.PASS_IN, None)
        return (o, )

class AkashicData:
    def __init__(self, *arg, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    def __str__(self) -> str:
        return {k: v for k, v in dir(self)}

class AkashicNode(JOVBaseNode):
    NAME = "AKASHIC (JOV) 📓"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Display the top level attributes of an output"
    RETURN_TYPES = (WILDCARD, 'AKASHIC', )
    RETURN_NAMES = (Lexicon.PASS_OUT, Lexicon.IO)
    OUTPUT_NODE = True
    SORT = 50

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PASS_IN: (WILDCARD, {})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def __parse(self, val) -> dict[str, list[Any]]:
        if isinstance(val, dict):
            result = "{"
            for k, v in val.items():
                result["text"] += f"{k}:{self.__parse(v)}, "
            return "text", [result[:-2] + "}"]
        elif isinstance(val, (tuple, set, list,)):
            result = "("
            for v in val:
                result += f"{self.__parse(v)}, "
            return "text", [result[:-2] + ")"]
        elif isinstance(val, str):
             return "text", [val]
        elif isinstance(val, bool):
            return "text", ["True" if val else "False"]
        elif isinstance(val, torch.Tensor):
            # logger.debug(f"Tensor: {val.shape}")
            ret = []
            if not isinstance(val, (list, tuple, set,)):
                val = [val]
            for img in val:
                img = tensor2pil(img)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img = base64.b64encode(buffered.getvalue())
                img = "data:image/png;base64," + img.decode("utf-8")
                ret.append(img)
            return "b64_images", ret
        else:
            # no clue what I am....
            meh = ''.join(repr(type(val)).split("'")[1:2])
            return "text", [meh]

    def run(self, **kw) -> tuple[Any, Any]:
        o = kw.get(Lexicon.PASS_IN, None)
        output = {"ui": {"b64_images": [], "text": []}}
        if o is None:
            output["ui"]["result"] = (o, {}, )
            return output

        for v in kw.values():
            who, data = self.__parse(v)
            output["ui"][who].extend(data)

        ak = AkashicData(image=output["ui"]["b64_images"], text=output["ui"]["text"] )
        output["result"] = (o, ak)
        return output

class ValueGraphNode(JOVBaseNode):
    NAME = "VALUE GRAPH (JOV) 📈"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Graphs historical execution run values"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    OUTPUT_NODE = True
    SORT = 100

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            # Lexicon.UNKNOWN: (WILDCARD, {"default": None}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
            Lexicon.VALUE: ("INT", {"default": 500, "min": 0})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __plot_parameter(self, data) -> None:
        ys = [data[x] for x in xs]
        #line = plt.plot(xs, ys, *args, **kw)
        line = plt.plot(xs, ys, label=data.label)
        kfx = data.keyframes
        kfy = [data[x] for x in kfx]
        plt.scatter(kfx, kfy, color=line[0].get_color())

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__history = []
        self.__index = 0
        self.__fig, self.__ax = plt.subplots(figsize=(11, 8))
        self.__ax.set_xlabel("FRAME")
        self.__ax.set_ylabel("VALUE")
        self.__ax.set_title("VALUE HISTORY")

    def run(self, **kw) -> tuple[torch.Tensor]:

        if kw.get(Lexicon.RESET, False):
            self.__history = []
            self.__index = 0

        elif not kw.get(Lexicon.WAIT, False):
            val = kw.get(Lexicon.UNKNOWN, 0)
            # logger.debug(val, type(val))
            if type(val) not in [bool, int, float, np.float16, np.float32, np.float64]:
                val = 0
            self.__history.append(val)
            self.__index += 1

        slice = kw.get(Lexicon.VALUE, 0)

        self.__ax.clear()

        print(kw)
        self.__ax.plot(self.__history[-slice + self.__index:], label=curve.label)

        self.__fig.canvas.draw_idle()

        buffer = io.BytesIO()
        self.__fig.savefig(buffer, format="png")
        buffer.seek(0)
        image = Image.open(buffer)
        return (pil2tensor(image),)

class RerouteNode(JOVBaseNode):
    NAME = "RE-ROUTE (JOV) 🚌"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Pass all data because the default is broken on connection"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.PASS_OUT, )
    SORT = 70

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PASS_IN: (WILDCARD, {})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[Any, Any]:
        o = kw.get(Lexicon.PASS_IN, None)
        return (o, )
