"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import io
import os
import time
import base64
import fnmatch
from typing import Any
import matplotlib.pyplot as plt

import torch
import numpy as np
from PIL import Image
from loguru import logger
from uuid import uuid4
from pathlib import Path

import comfy
from folder_paths import get_output_directory
from server import PromptServer
import nodes

from Jovimetrix import JOVBaseNode, IT_REQUIRED, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import deep_merge_dict
from Jovimetrix.sup.image import cv2tensor, image_load, tensor2pil, pil2tensor

# =============================================================================

class AkashicData:
    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
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

        logger.debug(kw)
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

class ExportNode(JOVBaseNode):
    NAME = "EXPORT (JOV) 📽"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = ""
    INPUT_IS_LIST = True
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PIXEL: ("IMAGE", ),
            Lexicon.PASS_OUT: ("STRING", {"default": get_output_directory()}),
            Lexicon.FORMAT: (["gif", "jpg", "png"], {"default": "png"}),
            Lexicon.OPTIMIZE: ("BOOLEAN", {"default": False}),
            Lexicon.FPS: ("INT", {"default": 0, "min": 0, "max": 120}),
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> None:
        img = kw.get(Lexicon.PIXEL, [])
        output_dir = kw.get(Lexicon.PASS_OUT, [])[0]
        format = kw.get(Lexicon.FORMAT, ["gif"])[0]
        optimize = kw.get(Lexicon.OPTIMIZE, [False])[0]
        fps = kw.get(Lexicon.FPS, [0])[0]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def output(extension) -> Path:
            return output_dir / f"{uuid4().hex[:16]}.{extension}"

        images = [tensor2pil(i).convert("RGB") for i in img]
        if format == "gif":
            images[0].save(
                output(format),
                append_images=images[1:],
                disposal=2,
                duration=1 / fps * 1000 if fps else 0,
                loop=0,
                optimize=optimize,
                save_all=True,
            )
        else:
            for img in images:
                img.save(output(format), optimize=optimize)

        return ()

class QueueNode(JOVBaseNode):
    NAME = "QUEUE (JOV) 🗃"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Cycle lists of images files or strings for node inputs."
    RETURN_TYPES = (WILDCARD, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.QUEUE: ("STRING", {"multiline": True, "default": ""}),
            # Lexicon.VALUE: ("INT", {"default": 0}),
            # -1 == HALT (send NONE), 0 = FOREVER, N-> COUNT TIMES THROUGH LIST
            Lexicon.LOOP: ("INT", {"default": 1, "min": 0}),
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
        },
        "hidden": {
            "id": "UNIQUE_ID"
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__loops = 0
        self.__index = 0
        self.__q = None

    def __parse(self, data) -> list:
        entries = []
        for line in data.strip().split('\n'):
            parts = [part.strip() for part in line.split(',')]

            count = 1
            try: count = int(parts[-1])
            except: pass

            path = Path(parts[0])
            data = [parts[0]]
            if path.is_dir():
                philter = parts[1] if len(parts) > 1 and isinstance(parts[1], str) else '*.png;*.jpg;*.webp'
                file_names = [file.name for file in path.iterdir() if file.is_file()]
                new_data = [path / fname for fname in file_names if any(fnmatch.fnmatch(fname, pat) for pat in philter.split(';'))]
                if len(new_data):
                    data = new_data
            if len(data) and count > 0:
                entries.extend(data * count)
        return entries

    def run(self, id, **kw) -> None:
        reset = kw.get(Lexicon.RESET, False)
        if reset:
            self.__q = None

        if self.__q is None:
            # process Q into ...
            # check if folder first, file, then string.
            # entry is: data, <filter if folder:*.png,*.jpg>, <repeats:1+>
            q = kw.get(Lexicon.QUEUE, "")
            self.__q = self.__parse(q)
            self.__loops = 0
            self.__index = 0
            self.__time = time.perf_counter()

        if self.__index >= len(self.__q):
            loop = kw.get(Lexicon.LOOP, 0)
            # we are done with X loops
            self.__loops += 1
            if loop > 0 and self.__loops >= loop:
                # hard halt?
                PromptServer.instance.send_sync("jovi-queue-done", {"id": id})
                nodes.interrupt_processing(True)
                t = time.perf_counter() - self.__time
                logger.warning(f"queue completed: {id} [{t}]")
                self.__q = None
                return ()
            self.__index = 0

        data = self.__q[self.__index]
        logger.info(f"QUEUE #{id} [{data}] ({self.__index}) |{self.__loops}|")
        PromptServer.instance.send_sync("jovi-queue-ping", {"id": id})
        self.__index += 1
        if os.path.isfile(data):
            data = cv2tensor(image_load(data)[0])
        return (data, )
