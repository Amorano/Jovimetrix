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
import shutil
from typing import Any
from pathlib import Path
from uuid import uuid4

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from loguru import logger

import comfy
from folder_paths import get_output_directory
from server import PromptServer
import nodes

from Jovimetrix import ComfyAPIMessage, JOVBaseNode, TimedOutException, \
    WILDCARD, ROOT, IT_REQUIRED, IT_WH, MIN_IMAGE_SIZE, IT_PIXEL, IT_PIXEL2

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import path_next, deep_merge_dict, parse_tuple, \
    zip_longest_fill
from Jovimetrix.sup.image import cv2mask, cv2tensor, tensor2pil, tensor2cv, \
    pil2tensor, image_load, image_formats, image_diff, channel_solid

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"

FORMATS = ["gif", "png", "jpg"]
if (JOV_GIFSKI := os.getenv("JOV_GIFSKI", None)) is not None:
    FORMATS = ["gifski"] + FORMATS
    logger.info("gifski support")
else:
    logger.warning("no gifski support")

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
    CATEGORY = JOV_CATEGORY
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
    CATEGORY = JOV_CATEGORY
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
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Pass all data because the default is broken on connection."
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = (Lexicon.PASS_OUT, )
    SORT = 5

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
    CATEGORY = JOV_CATEGORY
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

class ExportNode(JOVBaseNode):
    NAME = "EXPORT (JOV) 📽"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Take your frames out static or animated (GIF)"
    OUTPUT_NODE = True
    SORT = 80

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PREFIX: ("STRING", {"default": ""}),
            Lexicon.PASS_OUT: ("STRING", {"default": get_output_directory()}),
            Lexicon.OVERWRITE: ("BOOLEAN", {"default": False}),
            Lexicon.FORMAT: (FORMATS, {"default": FORMATS[0]}),
            # GIF ONLY
            Lexicon.OPTIMIZE: ("BOOLEAN", {"default": False}),
            # GIFSKI ONLY
            Lexicon.QUALITY: ("INT", {"default": 90, "min": 1, "max": 100}),
            Lexicon.QUALITY_M: ("INT", {"default": 100, "min": 1, "max": 100}),
            # GIF OR GIFSKI
            Lexicon.FPS: ("INT", {"default": 20, "min": 1, "max": 60}),
            # GIF OR GIFSKI
            Lexicon.LOOP: ("INT", {"default": 0, "min": 0}),
        }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL, d)

    def run(self, **kw) -> None:
        img = kw.get(Lexicon.PIXEL, [None])[0]
        suffix = kw.get(Lexicon.PREFIX, [""])[0]
        if suffix == "":
            suffix = uuid4().hex[:16]

        output_dir = kw.get(Lexicon.PASS_OUT, [""])[0]
        format = kw.get(Lexicon.FORMAT, ["gif"])[0]
        overwrite = kw.get(Lexicon.OVERWRITE, False)[0]
        optimize = kw.get(Lexicon.OPTIMIZE, [False])[0]
        quality = kw.get(Lexicon.QUALITY, [0])[0]
        motion = kw.get(Lexicon.QUALITY_M, [0])[0]
        fps = kw.get(Lexicon.FPS, [0])[0]
        loop = kw.get(Lexicon.LOOP, [0])[0]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def output(extension) -> Path:
            path = output_dir / f"{suffix}.{extension}"
            if not overwrite and os.path.isfile(path):
                path = str(output_dir / f"{suffix}_%s.{extension}")
                path = path_next(path)
            return path

        empty = Image.new("RGB", (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE))
        images = [tensor2pil(i) if i is not None else empty for i in img]
        images = [i.convert("RGB") if i is not None else empty for i in images]

        if format == "gifski":
            root = output_dir / f"{suffix}_{uuid4().hex[:16]}"
            try:
                root.mkdir(parents=True, exist_ok=True)
                for idx, i in enumerate(images):
                    fname = str(root / f"{suffix}_{idx}.png")
                    i.save(fname)
            except Exception as e:
                logger.warning(output_dir)
                logger.error(str(e))
                return

            out = output('gif')
            fps = f"--fps {fps}" if fps > 0 else ""
            q = f"--quality {quality}"
            mq = f"--motion-quality {motion}"
            cmd = f"{JOV_GIFSKI} -o {out} {q} {mq} {fps} {str(root)}/{suffix}_*.png"
            logger.info(cmd)
            try:
                os.system(cmd)
            except Exception as e:
                logger.warning(cmd)
                logger.error(str(e))

            shutil.rmtree(root)

        elif format == "gif":
            images[0].save(
                output('gif'),
                append_images=images[1:],
                disposal=2,
                duration=1 / fps * 1000 if fps else 0,
                loop=loop,
                optimize=optimize,
                save_all=True,
            )
        else:
            for img in images:
                img.save(output(format), optimize=optimize)

        return ()

class ImageDiffNode(JOVBaseNode):
    NAME = "IMAGE DIFF (JOV) 📏"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Explicitly show the differences between two images via self-similarity index."
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, True, True, )
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "FLOAT", )
    RETURN_NAMES = (Lexicon.IN_A, Lexicon.IN_B, Lexicon.DIFF, Lexicon.THRESHOLD, Lexicon.FLOAT, )
    SORT = 90

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
              Lexicon.THRESHOLD: ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01},),
        }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL2, d)

    def run(self, **kw) -> tuple[Any, Any]:
        a = kw.get(Lexicon.PIXEL_A, [None])
        b = kw.get(Lexicon.PIXEL_B, [None])
        th = kw.get(Lexicon.THRESHOLD, [0])
        image_a = []
        image_b = []
        diff = []
        thresh = []
        score = []
        params = [tuple(x) for x in zip_longest_fill(a, b, th)]
        if len(params) == 0:
            e = [torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu")]
            m = [torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=torch.uint8, device="cpu")]
            return e, e, m, m, [1.],

        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (a, b, th) in enumerate(params):
            if a is None and b is None:
                e = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu")
                image_a.append(e)
                image_b.append(e)
                m = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=torch.uint8, device="cpu")
                diff.append(m)
                thresh.append(m)
                score.append(1.)
                continue

            width = MIN_IMAGE_SIZE
            height = MIN_IMAGE_SIZE
            _, height, width, _ = a.shape if a is not None else b.shape
            a = tensor2cv(a) if a is not None else channel_solid(width, height, 0)
            b = tensor2cv(b) if b is not None else channel_solid(width, height, 0)

            a, b, d, t, s = image_diff(a, b, int(th * 255.))
            image_a.append(cv2tensor(a))
            image_b.append(cv2tensor(b))
            diff.append(cv2mask(d))
            thresh.append(cv2mask(t))
            score.append(s)
            pbar.update_absolute(idx)

        return image_a, image_b, diff, thresh, score,
