"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import io
import os
import json
import glob
import base64
from pickle import FALSE
import random
from typing import Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from loguru import logger

from server import PromptServer
import nodes

from Jovimetrix import ComfyAPIMessage, JOVBaseNode, TimedOutException, \
    IT_REQUIRED, WILDCARD, ROOT, IT_WH, MIN_IMAGE_SIZE

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import deep_merge_dict, parse_tuple
from Jovimetrix.sup.image import cv2mask, cv2tensor, image_load, tensor2pil, \
    pil2tensor, image_formats

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
    DESCRIPTION = "Display the top level attributes of an output."
    RETURN_TYPES = (WILDCARD, 'AKASHIC', )
    RETURN_NAMES = (Lexicon.PASS_OUT, Lexicon.IO)
    OUTPUT_NODE = True
    SORT = 10

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
    DESCRIPTION = "Graphs historical execution run values."
    INPUT_IS_LIST = False
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    OUTPUT_NODE = True
    SORT = 15

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
            Lexicon.VALUE: ("INT", {"default": 120, "min": 0})
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WH)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__history = []
        self.__index = [0]
        self.__fig, self.__ax = plt.subplots(figsize=(5.12, 3.72))
        self.__ax.set_xlabel("FRAME")
        self.__ax.set_ylabel("VALUE")
        self.__ax.set_title("VALUE HISTORY")

    def run(self, **kw) -> tuple[torch.Tensor]:

        reset = kw.get(Lexicon.RESET, False)
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

        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        width, height = wihi
        wihi = (width / 100., height / 100.)
        if self.__fig.figsize() != wihi:
            self.__fig.set_figwidth(wihi[0])
            self.__fig.set_figheight(wihi[1])

        self.__fig.canvas.draw_idle()
        buffer = io.BytesIO()
        self.__fig.savefig(buffer, format="png")
        buffer.seek(0)
        image = Image.open(buffer)
        return (pil2tensor(image),)

class RerouteNode(JOVBaseNode):
    NAME = "RE-ROUTE (JOV) 🚌"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Pass all data because the default is broken on connection."
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.PASS_OUT, )
    SORT = 100

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PASS_IN: (WILDCARD, {})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[Any, Any]:
        o = kw.get(Lexicon.PASS_IN, None)
        return (o, )

class QueueNode(JOVBaseNode):
    NAME = "QUEUE (JOV) 🗃"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Cycle lists of images files or strings for node inputs."
    INPUT_IS_LIST = False
    RETURN_TYPES = (WILDCARD, "MASK", WILDCARD, "STRING", "INT", "INT", )
    RETURN_NAMES = (Lexicon.ANY, Lexicon.MASK, Lexicon.QUEUE, Lexicon.CURRENT, Lexicon.VALUE, Lexicon.TOTAL, )
    OUTPUT_IS_LIST = (True, True, True, True, True, True, )
    VIDEO_FORMATS = ['.webm', '.mp4', '.avi', '.wmv', '.mkv', '.mov', '.mxf']
    SORT = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.QUEUE: ("STRING", {"multiline": True, "default": ""}),
                Lexicon.LOOP: ("INT", {"default": 0, "min": 0}),
                Lexicon.RANDOM: ("BOOLEAN", {"default": False}),
                Lexicon.BATCH: ("INT", {"default": 1, "min": 1}),
                Lexicon.BATCH_LIST: ("BOOLEAN", {"default": True}),
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
        batch_list = kw.get(Lexicon.BATCH_LIST, True)
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

        return [data] * batch, [mask] * batch, [self.__q] * batch, [current] * batch, [self.__index] * batch, [self.__len] * batch,
