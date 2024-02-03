"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""


import io
import os
import json
import glob
import base64
import random
from enum import Enum
from typing import Any
from uuid import uuid4
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import numpy as np
from PIL import Image
from loguru import logger

import comfy
from folder_paths import get_output_directory
from server import PromptServer
import nodes

from Jovimetrix import ComfyAPIMessage, JOVBaseNode, TimedOutException, \
    TYPE_IMAGE, IT_REQUIRED, WILDCARD, ROOT

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import deep_merge_dict, zip_longest_fill

from Jovimetrix.sup.image import cv2mask, cv2tensor, image_load, tensor2pil, \
    pil2tensor, image_formats

from Jovimetrix.sup.anim import Ease, EnumEase

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

class EnumConvertType(Enum):
    STRING = 0
    BOOLEAN = 10
    INT = 20
    FLOAT   = 30
    VEC2 = 40
    VEC3 = 50
    VEC4 = 60
    # TUPLE = 70

class ConvertNode(JOVBaseNode):
    """Convert A to B."""
    NAME = "CONVERT (JOV) 🧬"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Convert A to B."
    RETURN_TYPES = (WILDCARD,)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    SORT = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.IN_A: (WILDCARD, {"default": None}),
            Lexicon.TYPE: (["STRING", "BOOLEAN", "INT", "FLOAT", "VEC2", "VEC3", "VEC4"], {"default": "BOOLEAN"})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    @staticmethod
    def convert(typ, val) -> tuple | tuple[Any]:
        size = len(val) if type(val) == tuple else 0
        if typ in ["STRING"]:
            if size > 0:
                return " ".join(str(val))
            return str(val)
        elif typ in ["FLOAT"]:
            if size > 0:
                return float(val[0])
            return float(val)
        elif typ == "BOOLEAN":
            if size > 0:
                return bool(val[0])
            return bool(val)
        elif typ == "INT":
            if size > 0:
                return int(val[0])
            return int(val)
        elif typ == "VEC2":
            if size > 1:
                return (val[0], val[1], )
            elif size > 0:
                return (val[0], val[0], )
            return (val, val, )
        elif typ == "VEC3":
            if size > 2:
                return (val[0], val[1], val[2], )
            elif size > 1:
                return (val[0], val[1], val[1], )
            elif size > 0:
                return (val[0], val[0], val[0], )
            return (val, val, val, )
        elif typ == "VEC4":
            if size > 3:
                return (val[0], val[1], val[2], val[3], )
            elif size > 2:
                return (val[0], val[1], val[2], val[2], )
            elif size > 1:
                return (val[0], val[1], val[1], val[1], )
            elif size > 0:
                return (val[0], val[0], val[0], val[0], )
            return (val, val, val, val, )
        else:
            return "nan"

    def run(self, **kw) -> tuple[bool]:
        results = []
        typ = kw.pop(Lexicon.TYPE, ["STRING"])
        values = kw.values()
        params = [tuple(x) for x in zip_longest_fill(typ, values)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (typ, values) in enumerate(params):
            result = []

            v = ''
            try: v = next(iter(values))
            except: pass

            if not isinstance(v, (list, set, tuple)):
                v = [v]

            for idx, val in enumerate(v):
                val_new = ConvertNode.convert(typ, val)
                result.append(val_new)

            results.append(result)
            pbar.update_absolute(idx)

        return results

class ValueNode(JOVBaseNode):
    """Create a value for most types."""

    NAME = "VALUE (JOV) #️⃣"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Create a value for most types; also universal constants."
    INPUT_IS_LIST = True
    RETURN_TYPES = (WILDCARD, )
    OUTPUT_IS_LIST = (True, )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.TYPE: (EnumConvertType._member_names_, {"default": EnumConvertType.BOOLEAN.name}),
                Lexicon.X: ("FLOAT", {"default": 0}),
                Lexicon.Y: ("FLOAT", {"default": 0}),
                Lexicon.Z: ("FLOAT", {"default": 0}),
                Lexicon.W: ("FLOAT", {"default": 0})
            }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        typ = kw.get(Lexicon.TYPE, [EnumConvertType.BOOLEAN])
        x = kw.get(Lexicon.X, [None])
        y = kw.get(Lexicon.Y, [0])
        z = kw.get(Lexicon.Z, [0])
        w = kw.get(Lexicon.W, [0])
        params = [tuple(x) for x in zip_longest_fill(typ, x, y, z, w)]
        logger.debug(params)
        results = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (typ, x, y, z, w) in enumerate(params):
            typ = EnumConvertType[typ]
            if typ == EnumConvertType.STRING:
                results.append("" if x is None else str(x))
                continue

            x = 0 if x is None else x
            match typ:
                case EnumConvertType.VEC2:
                    results.append((x, y,))
                case EnumConvertType.VEC3:
                    results.append((x, y, z,))
                case EnumConvertType.VEC4:
                    results.append((x, y, z, w,))
                case _:
                    results.append(x)

            pbar.update_absolute(idx)
        logger.debug(results)
        return (results, )

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
        self.__index = [0]
        self.__fig, self.__ax = plt.subplots(figsize=(11, 8))
        self.__ax.set_xlabel("FRAME")
        self.__ax.set_ylabel("VALUE")
        self.__ax.set_title("VALUE HISTORY")

    def run(self, **kw) -> tuple[torch.Tensor]:

        if kw.get(Lexicon.RESET, False):
            self.__history = [[]]
            self.__index = [0]

        idx = 1
        while 1:
            who = f"{Lexicon.UNKNOWN}_{idx}"
            if (val := kw.get(who, None)) is None:
                break
            if type(val) not in [bool, int, float, np.float16, np.float32, np.float64]:
                val = 0

            while len(self.__history) < idx:
                self.__history.append([])
                self.__index.append(0)
            self.__history[idx-1].append(val)
            idx += 1

        slice = kw.get(Lexicon.VALUE, 0)
        self.__ax.clear()
        for i, h in enumerate(self.__history):
            self.__ax.plot(h[max(0, -slice + self.__index[i]):], color="rgbcymk"[i])
            # self.__ax.scatter(kfx, kfy, color=line[0].get_color())
            self.__index[i] += 1

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
    RETURN_TYPES = (WILDCARD, "MASK", WILDCARD, "INT", "STRING", "INT", )
    RETURN_NAMES = (Lexicon.ANY, Lexicon.MASK, Lexicon.QUEUE, Lexicon.COUNT, Lexicon.CURRENT, Lexicon.VALUE, )
    OUTPUT_IS_LIST = (True, True, True, True, True, True, )
    VIDEO_FORMATS = ['.webm', '.mp4', '.avi', '.wmv', '.mkv', '.mov', '.mxf']

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.QUEUE: ("STRING", {"multiline": True, "default": ""}),
                Lexicon.LOOP: ("INT", {"default": 0, "min": 0}),
                Lexicon.RANDOM: ("BOOLEAN", {"default": False}),
                Lexicon.BATCH: ("INT", {"default": 1, "min": 1}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
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
        self.__q_rand = None
        self.__last = None
        self.__len = 0
        self.__previous = None
        self.__previous_mask = None

    def __parse(self, data) -> list:
        entries = []
        for line in data.strip().split('\n'):
            parts = [part.strip() for part in line.split(',')]

            count = 1
            try: count = int(parts[-1])
            except: pass

            data = [parts[0]]
            path = Path(parts[0])
            path2 = Path(ROOT / parts[0])
            if path.is_dir() or path2.is_dir():
                philter = parts[1].split(';') if len(parts) > 1 and isinstance(parts[1], str) else image_formats()
                philter.extend(self.VIDEO_FORMATS)
                path = path if path.is_dir() else path2
                file_names = [file.name for file in path.iterdir() if file.is_file()]
                new_data = [str(path / fname) for fname in file_names if any(fname.endswith(pat) for pat in philter)]
                if len(new_data):
                    data = new_data
            elif path.is_file() or path2.is_file():
                path = path if path.is_file() else path2
                data = [str(path.resolve())]
            elif len(results := glob.glob(str(path2))) > 0:
                data = [x.replace('\\\\', '/') for x in results]

            if len(data) and count > 0:
                entries.extend(data * count)
        return entries

    def run(self, id, **kw) -> None:

        def process(data: str) -> tuple[torch.Tensor, torch.Tensor] | str | dict:
            mask = None
            if not os.path.isfile(data):
                return data, mask
            #try:
            _, ext = os.path.splitext(data)
            if ext in image_formats():
                data, mask = image_load(data)
                data = cv2tensor(data)
                mask = cv2mask(mask)
            elif ext == '.json':
                with open(data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif ext == '.txt':
                with open(data, 'r', encoding='utf-8') as f:
                    data = f.read()
            #except Exception as e:
            #    logger.error(data)
            #    logger.error(str(e))
            return data, mask

        reset = kw.get(Lexicon.RESET, False)
        rand = kw.get(Lexicon.RANDOM, False)

        # clear the queue of msgs...
        # better resets? check if reset message
        try:
            data = ComfyAPIMessage.poll(id, timeout=0)
            # logger.debug(data)
            if (cmd := data.get('cmd', None)) is not None:
                if cmd == 'reset':
                    reset = True
        except TimedOutException as e:
            pass
        except Exception as e:
            logger.error(str(e))

        if reset:
            self.__q = None
            self.__q_rand = None

        if self.__q is None:
            # process Q into ...
            # check if folder first, file, then string.
            # entry is: data, <filter if folder:*.png,*.jpg>, <repeats:1+>
            q = kw.get(Lexicon.QUEUE, "")
            self.__q = self.__parse(q)
            self.__q_rand = list(self.__q)
            random.shuffle(self.__q_rand)
            self.__len = len(self.__q) - 1
            self.__loops = 0
            self.__index = 0
            self.__last = 0
            self.__previous = self.__q[0] if len(self.__q) else None
            if self.__previous:
                self.__previous, self.__previous_mask = process(self.__previous)

        if (wait := kw.get(Lexicon.WAIT, False)):
            self.__index = self.__last

        if self.__index >= len(self.__q):
            loop = kw.get(Lexicon.LOOP, 0)
            # we are done with X loops
            self.__loops += 1
            if loop > 0 and self.__loops >= loop:
                # hard halt?
                PromptServer.instance.send_sync("jovi-queue-done", {"id": id})
                nodes.interrupt_processing(True)
                logger.warning(f"Q Complete [{id}]")
                self.__q = None
                self.__q_rand = None
                return ()

            random.shuffle(self.__q_rand)
            self.__index = 0

        if rand:
            current = self.__q_rand[self.__index]
        else:
            current = self.__q[self.__index]
        info = f"QUEUE #{id} [{current}] ({self.__index})"

        if self.__loops:
            info += f" |{self.__loops}|"

        if wait:
            info += f" PAUSED"

        data = self.__previous
        mask = self.__previous_mask
        batch = max(1, kw.get(Lexicon.BATCH, 1))
        if not wait:
            if rand:
                data, mask = process(self.__q_rand[self.__index])
            else:
                data, mask = process(self.__q[self.__index])
            # data = [data]
            # mask = [mask]
            self.__index += 1

        self.__last = self.__index
        self.__previous = data
        self.__previous_mask = mask
        PromptServer.instance.send_sync("jovi-queue-ping", {"id": id, "c": current, "i": self.__index, "s": self.__len, "l": self.__q})

        return [data] * batch, [mask] * batch, [self.__q] * batch, [self.__len] * batch, [current] * batch, [self.__index] * batch,

class EnumNumberType(Enum):
    INT = 0
    FLOAT = 10

class LerpNode(JOVBaseNode):
    NAME = "LERP (JOV) 📏"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Interpolate between two values with or without a smoothing."
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.ANY )
    SORT = 90

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.LERP_A: (WILDCARD, {}),
            Lexicon.LERP_B: (WILDCARD, {}),
            Lexicon.FLOAT: ("FLOAT", {"default": 0., "min": 0., "max": 1.0, "step": 0.001, "precision": 4, "round": 0.00001}),
            Lexicon.EASE: (["NONE"] + EnumEase._member_names_, {"default": "NONE"}),
            Lexicon.TYPE: (EnumNumberType._member_names_, {"default": EnumNumberType.FLOAT.name})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[Any, Any]:
        a = kw.get(Lexicon.LERP_A, [0])
        b = kw.get(Lexicon.LERP_B, [0])
        pos = kw.get(Lexicon.FLOAT, [0.])
        op = kw.get(Lexicon.EASE, ["NONE"])
        typ = kw.get(Lexicon.TYPE, ["NONE"])

        value = []
        params = [tuple(x) for x in zip_longest_fill(a, b, pos, op, typ)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (a, b, pos, op, typ) in enumerate(params):
            # make sure we only interpolate between the smallest "stride" we can
            size = min(len(a), len(b))
            typ = EnumNumberType[typ]
            ease = EnumEase[op]

            def same():
                val = 0.
                if op == "NONE":
                    val = b * pos + a * (1 - pos)
                else:
                    val = Ease.ease(ease, start=a, end=b, alpha=pos)
                return val

            if size == 3:
                same()
            elif size == 2:
                same()
            elif size == 1:
                same()

            pbar.update_absolute(idx)
        return (value, )
