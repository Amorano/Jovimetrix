"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import io
import os
import json
import glob
import random
from enum import Enum
from uuid import uuid4
from pathlib import Path
from itertools import zip_longest
from typing import Any, Tuple

import torch
from PIL import Image
import matplotlib.pyplot as plt
from loguru import logger

from comfy.utils import ProgressBar
from folder_paths import get_output_directory

from Jovimetrix import comfy_message, parse_reset, JOVBaseNode, \
    WILDCARD, ROOT, JOV_WEB_RES_ROOT

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_dynamic, path_next, \
    parse_param, zip_longest_fill, EnumConvertType

from Jovimetrix.sup.image import channel_solid, cv2pil, \
    cv2tensor, image_convert, tensor2cv, pil2tensor, image_load, image_formats, \
    image_diff, EnumImageType, MIN_IMAGE_SIZE

# =============================================================================

JOV_CATEGORY = "UTILITY"

FORMATS = ["gif", "png", "jpg"]
if (JOV_GIFSKI := os.getenv("JOV_GIFSKI", None)) is not None:
    if not os.path.isfile(JOV_GIFSKI):
        logger.error(f"gifski missing [{JOV_GIFSKI}]")
        JOV_GIFSKI = None
    else:
        FORMATS = ["gifski"] + FORMATS
        logger.info("gifski support")
else:
    logger.warning("no gifski support")

class EnumBatchMode(Enum):
    MERGE = 30
    PICK = 10
    SLICE = 15
    INDEX_LIST = 20
    RANDOM = 5

# =============================================================================

class AkashicData:
    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

class AkashicNode(JOVBaseNode):
    NAME = "AKASHIC (JOV) 📓"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {},
            "optional": {
                Lexicon.PASS_IN: (WILDCARD, {})
            }
        }
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> Tuple[Any, Any]:
        o = parse_param(kw, Lexicon.PASS_IN, EnumConvertType.ANY, None)
        output = {"ui": {"b64_images": [], "text": []}}
        if o is None:
            output["ui"]["result"] = (None, None, )
            return output

        def __parse(val) -> str:
            ret = val
            typ = ''.join(repr(type(val)).split("'")[1:2])
            if isinstance(val, dict):
                ret = ', '.join(f"{k}:{v}" for k, v in val.items())
                return f"{{{ret}}} [{typ}]"
            elif isinstance(val, (tuple, set, list,)):
                if len(val) < 2:
                    ret = val[0]
                    typ = ''.join(repr(type(ret)).split("'")[1:2])
                    return f"{ret} [{typ}]"
                ret = ', '.join(str(v) for v in val)
                return f"({ret}) [{typ}]"
            elif isinstance(val, (int, float, str)):
                return f"{val} [{typ}]"
            elif isinstance(val, bool):
                val = "True" if val else "False"
                return f"{val} [{typ}]"
            elif isinstance(val, torch.Tensor):
                ret = []
                if not isinstance(val, (list, tuple, set,)):
                    val = [val]
                for img in val:
                    #img = tensor2pil(img)
                    ret.append(f"({'x'.join([str(x) for x in img.shape])}) [{typ}]")
                    #ret += str(img.size)
                    #buffered = io.BytesIO()
                    #img.save(buffered, format="PNG")
                    #img = base64.b64encode(buffered.getvalue())
                    #img = "data:image/png;base64," + img.decode("utf-8")
                    #output["ui"]["b64_images"].append(img)
                return ', '.join(ret)
            return f"unknown [{typ}]"

        for x in o:
            output["ui"]["text"].append(__parse(x))
        #ak = AkashicData(image=output["ui"]["b64_images"], text=output["ui"]["text"] )
        #output["result"] = (o, ak)
        return output

class ValueGraphNode(JOVBaseNode):
    NAME = "GRAPH (JOV) 📈"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    SORT = 15

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
            Lexicon.VALUE: ("INT", {"default": 60, "min": 0, "tooltip":"Number of values to graph and display"}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]})
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__history = []
        self.__fig, self.__ax = plt.subplots(figsize=(5.12, 5.12))

    def run(self, ident, **kw) -> Tuple[torch.Tensor]:
        slice = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 60)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), 1)[0]
        if parse_reset(ident) > 0 or parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]:
            self.__history = []
        longest_edge = 0
        dynamic = parse_dynamic(kw, Lexicon.UNKNOWN, EnumConvertType.FLOAT, 0)
        for idx, val in enumerate(dynamic):
            if isinstance(val, (set, tuple,)):
                val = list(val)
            if not isinstance(val, (list, )):
                val = [val]
            while len(self.__history) <= idx:
                self.__history.append([])
            self.__history[idx].extend(val)
            stride = max(0, -slice + len(self.__history[idx]) + 1)
            longest_edge = max(longest_edge, stride)
            self.__history[idx] = self.__history[idx][stride:]
        self.__history = self.__history[:idx+1]
        self.__ax.clear()
        for i, h in enumerate(self.__history):
            self.__ax.plot(h, color="rgbcymk"[i])

        width, height = wihi
        wihi = (width / 100., height / 100.)
        self.__fig.set_figwidth(wihi[0])
        self.__fig.set_figheight(wihi[1])
        self.__fig.canvas.draw_idle()
        buffer = io.BytesIO()
        self.__fig.savefig(buffer, format="png")
        buffer.seek(0)
        image = Image.open(buffer)
        return (pil2tensor(image),)

class QueueNode(JOVBaseNode):
    NAME = "QUEUE (JOV) 🗃"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = (WILDCARD, WILDCARD, "STRING", "INT", "INT", )
    RETURN_NAMES = (Lexicon.ANY, Lexicon.QUEUE, Lexicon.CURRENT, Lexicon.INDEX, Lexicon.TOTAL, )
    VIDEO_FORMATS = ['.webm', '.mp4', '.avi', '.wmv', '.mkv', '.mov', '.mxf']
    SORT = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.QUEUE: ("STRING", {"multiline": True, "default": ""}),
            Lexicon.VALUE: ("INT", {"min": 0, "default": 0, "step": 1, "tooltip": "the current index for the current queue item"}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False, "tooltip":"Hold the item at the current queue index"}),
            Lexicon.RESET: ("BOOLEAN", {"default": False, "tooltip":"reset the queue back to index 1"}),
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self) -> None:
        self.__index = 0
        self.__q = None
        self.__index_last = None
        self.__len = 0
        self.__previous = None
        self.__last_q_value = {}

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
                path = str(path.resolve())
                if path.lower().endswith('.txt'):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = f.read().split('\n')
                else:
                    data = [path]
            elif len(results := glob.glob(str(path2))) > 0:
                data = [x.replace('\\\\', '/') for x in results]

            if len(data) and count > 0:
                entries.extend(data * count)
        return entries

    def run(self, ident, **kw) -> None:

        def process(q_data: str) -> Tuple[torch.Tensor, torch.Tensor] | str | dict:
            # single Q cache to skip loading single entries over and over
            if (val := self.__last_q_value.get(q_data, None)) is not None:
                return val
            if not os.path.isfile(q_data):
                return q_data
            _, ext = os.path.splitext(q_data)
            if ext in image_formats():
                data = image_load(q_data)[0]
                if len(data.shape) == 3:
                    h, w, cc = data.shape
                    data = cv2tensor(data)
                    if cc == 1:
                        data = data.unsqueeze(-1)
                self.__last_q_value[q_data] = data
            elif ext == '.json':
                with open(q_data, 'r', encoding='utf-8') as f:
                    self.__last_q_value[q_data] = json.load(f)
            return self.__last_q_value.get(q_data, q_data)

        # should work headless as well
        if parse_reset(ident) > 0 or parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]:
            self.__q = None
            self.__index = 0

        if (new_val := parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, self.__index))[0] > 0:
            self.__index = new_val

        if self.__q is None:
            # process Q into ...
            # check if folder first, file, then string.
            # entry is: data, <filter if folder:*.png,*.jpg>, <repeats:1+>
            q = parse_param(kw, Lexicon.QUEUE, EnumConvertType.STRING, "")[0]
            self.__q = self.__parse(q)
            self.__len = len(self.__q)
            self.__index_last = 0
            self.__previous = self.__q[0] if len(self.__q) else None
            if self.__previous:
                self.__previous = process(self.__previous)

        if (wait := parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False))[0] == True:
            self.__index = self.__index_last

        self.__index = max(0, self.__index) % self.__len
        current = self.__q[self.__index]
        data = self.__previous
        self.__index_last = self.__index
        info = f"QUEUE #{ident} [{current}] ({self.__index})"
        if wait == True:
            info += f" PAUSED"
        else:
            data = process(self.__q[self.__index])
            self.__index += 1

        self.__previous = data
        msg = {"id": ident,
               "c": current,
               "i": self.__index_last+1,
               "s": self.__len,
               "l": self.__q
        }
        comfy_message(ident, "jovi-queue-ping", msg)
        return data, self.__q, current, self.__index_last+1, self.__len

class ExportNode(JOVBaseNode):
    NAME = "EXPORT (JOV) 📽"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    SORT = 80

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.PASS_OUT: ("STRING", {"default": get_output_directory()}),
            Lexicon.FORMAT: (FORMATS, {"default": FORMATS[0]}),
            Lexicon.PREFIX: ("STRING", {"default": "jovi"}),
            Lexicon.OVERWRITE: ("BOOLEAN", {"default": False}),
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
        return Lexicon._parse(d, cls.HELP_URL)
    SORT = 2000

    def run(self, **kw) -> None:
        images = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        suffix = parse_param(kw, Lexicon.PREFIX, EnumConvertType.STRING, uuid4().hex[:16])[0]
        output_dir = parse_param(kw, Lexicon.PASS_OUT, EnumConvertType.STRING, "")[0]
        format = parse_param(kw, Lexicon.FORMAT, EnumConvertType.STRING, "gif")[0]
        overwrite = parse_param(kw, Lexicon.OVERWRITE, EnumConvertType.BOOLEAN, False)[0]
        optimize = parse_param(kw, Lexicon.OPTIMIZE, EnumConvertType.BOOLEAN, False)[0]
        quality = parse_param(kw, Lexicon.QUALITY, EnumConvertType.INT, 100, 0, 1)[0]
        motion = parse_param(kw, Lexicon.QUALITY_M, EnumConvertType.INT, 100, 0, 1)[0]
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 60, 1, 24)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0, 0)[0]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def output(extension) -> Path:
            path = output_dir / f"{suffix}.{extension}"
            if not overwrite and os.path.isfile(path):
                path = str(output_dir / f"{suffix}_%s.{extension}")
                path = path_next(path)
            return path

        images = [tensor2cv(i) for i in images]
        empty = Image.new("RGB", (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE))
        images = [cv2pil(i) for i in images]
        if format == "gifski":
            root = output_dir / f"{suffix}_{uuid4().hex[:16]}"
            # logger.debug(root)
            try:
                root.mkdir(parents=True, exist_ok=True)
                for idx, i in enumerate(images):
                    fname = str(root / f"{suffix}_{idx}.png")
                    i.save(fname)
            except Exception as e:
                logger.warning(output_dir)
                logger.error(str(e))
                return
            else:
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

                # shutil.rmtree(root)

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
        images = [pil2tensor(r) for r in images]
        return [torch.stack(images, dim=0).squeeze(1)]

class ImageDiffNode(JOVBaseNode):
    NAME = "IMAGE DIFF (JOV) 📏"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", ) #"FLOAT", )
    RETURN_NAMES = (Lexicon.IN_A, Lexicon.IN_B, Lexicon.DIFF, Lexicon.THRESHOLD) #, Lexicon.FLOAT, )
    SORT = 90

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.PIXEL_B: (WILDCARD, {}),
            Lexicon.THRESHOLD: ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> Tuple[Any, Any]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        th = parse_param(kw, Lexicon.THRESHOLD, EnumConvertType.FLOAT, 0.5, 0, 1)
        print(th)
        results = []
        params = list(zip_longest_fill(pA, pB, th))
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, th) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor2cv(pA)
            pB = channel_solid(chan=EnumImageType.BGRA) if pB is None else tensor2cv(pB)
            print(th)
            a, b, d, t, s = image_diff(pA, pB, int(th * 255))
            d = image_convert(d, 1)
            t = image_convert(t, 1)
            results.append([cv2tensor(a), cv2tensor(b), cv2tensor(d), cv2tensor(t)])
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*results))]
        return [list(a) for a in zip(*results)]

class ArrayNode(JOVBaseNode):
    NAME = "ARRAY (JOV) 📚"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("INT", WILDCARD, WILDCARD,)
    RETURN_NAMES = (Lexicon.VALUE, Lexicon.ANY, Lexicon.LIST,)
    SORT = 50

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.BATCH_MODE: (EnumBatchMode._member_names_, {"default": EnumBatchMode.MERGE.name, "tooltip":"select a single index, specific range, custom index list or randomized"}),
            Lexicon.INDEX: ("INT", {"default": 0, "min": 0, "step": 1}),
            Lexicon.RANGE: ("VEC3", {"default": (0, 0, 1)}),
            Lexicon.STRING: ("STRING", {"default": ""}),
            Lexicon.SEED: ("INT", {"default": 0, "min": 0, "step": 1}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
            Lexicon.BATCH_CHUNK: ("INT", {"default": 0, "min": 0, "step": 1}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    @classmethod
    def batched(cls, iterable, chunk_size, expand:bool=False, fill:Any=None) -> list:
        if expand:
            iterator = iter(iterable)
            return zip_longest(*[iterator] * chunk_size, fillvalue=fill)
        return [iterable[i: i + chunk_size] for i in range(0, len(iterable), chunk_size)]
        # return iter(lambda: tuple(islice(iterator, chunk_size)), tuple())

    def run(self, **kw) -> Tuple[int, list]:
        batch = parse_dynamic(Lexicon.UNKNOWN, kw)
        mode = parse_param(kw, Lexicon.BATCH_MODE, EnumConvertType.STRING, EnumBatchMode.MERGE.name)[0]
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)[0]
        chunk = parse_param(kw, Lexicon.BATCH_CHUNK, EnumConvertType.INT, 0)[0]
        extract = []
        # track latents since they need to be added back to Dict['samples']
        latents = []
        full = []
        for b in batch:
            if isinstance(b, dict) and "samples" in b:
                # latents are batched in the x.samples key
                data = b["samples"]
                full.extend([{"samples": [i]} for i in data])
                extract.extend(data)
                latents.extend([True] * len(data))
            elif isinstance(b, torch.Tensor):
                data = [i for i in batch]
                full.extend(data)
                extract.extend(data)
                latents.extend([False] * len(data))
            elif isinstance(b, (list, set, tuple,)):
                full.extend(b)
                extract.extend(b)
                latents.extend([False] * len(b))
            else:
                full.append(b)
                extract.append(b)
                latents.append(False)

        if mode == EnumBatchMode.PICK:
            index = parse_param(kw, Lexicon.BATCH_CHUNK, EnumConvertType.INT, 0, 0)
            index = index if index < len(extract) else -1
            extract = [extract[index]]
            if latents[index]:
                extract = {"samples": extract}
        elif mode == EnumBatchMode.SLICE:
            slice_range = parse_param(kw, Lexicon.RANGE, EnumConvertType.VEC3INT, (0, 0, 1))[0]
            start, end, step = slice_range
            end = len(extract) if end == 0 else end
            data = extract[start:end:step]
            latents = latents[start:end:step]
            extract = []
            for i, lat in enumerate(latents):
                dat = data[i]
                if lat:
                    dat = {"samples": [dat]}
                extract.append(dat)
        elif mode == EnumBatchMode.RANDOM:
            seed = parse_param(kw, Lexicon.SEED, EnumConvertType.INT, 0)
            random.seed(seed)
            full = random.choices(full)
            idx = random.randrange(0, len(extract))
            extract = [extract[idx]]
            if latents[idx]:
                extract = {"samples": extract}
        elif mode == EnumBatchMode.INDEX_LIST:
            indices = parse_param(kw, Lexicon.STRING, EnumConvertType.STRING, "")
            data = [extract[i:j] for i, j in zip([0]+indices, indices+[None])]
            latents = [latents[i:j] for i, j in zip([0]+indices, indices+[None])]
            extract = []
            for i, lat in enumerate(latents):
                dat = data[i]
                if lat:
                    dat = {"samples": [dat]}
                extract.append(dat)

        if flip and len(extract) > 1:
            extract = extract[::-1]

        if chunk > 0:
            extract = [e for e in self.batched(extract, chunk)]
        return (len(extract), extract, full,)

'''
class RESTNode:
    """Make requests and process the responses."""
    NAME = "REST (JOV) 😴"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("JSON", "INT", "STRING")
    RETURN_NAMES = ("RESPONSE", "LENGTH", "TOKEN")
    SORT = 80

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.API: ("STRING", {"default": ""}),
            Lexicon.URL: ("STRING", {"default": ""}),
            Lexicon.ATTRIBUTE: ("STRING", {"default": ""}),
            Lexicon.AUTH: ("STRING", {"multiline": True, "dynamic": False}),
            Lexicon.PATH: ("STRING", {"default": ""}),
            "iteration_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def authenticate(self, auth_url, auth_body, token_attribute_name):
        try:
            response = requests.post(auth_url, json=auth_body)
            response.raise_for_status()
            return response.json().get(token_attribute_name)
        except requests.exceptions.RequestException as e:
            logger.error(f"error obtaining bearer token - {e}")

    def run(self, **kw):
        auth_body_text = parse_param(kw, Lexicon.AUTH, EnumConvertType.STRING, "")
        api_url = parse_param(kw, Lexicon.URL, EnumConvertType.STRING, "")
        attribute = parse_param(kw, Lexicon.ATTRIBUTE, EnumConvertType.STRING, "")
        array_path = parse_param(kw, Lexicon.PATH, EnumConvertType.STRING, "")
        results = []
        params = list(zip_longest_fill(auth_body_text, api_url, attribute, array_path))
        pbar = ProgressBar(len(params))
        for idx, (auth_body_text, api_url, attribute, array_path) in enumerate(params):
            auth_body = None
            if auth_body_text:
                try:
                    auth_body = json.loads("{" + auth_body_text + "}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON input: {e}")
                    results.append([None, None, None])
                    pbar.update_absolute(idx)
                    continue

            headers = {}
            if api_url:
                token = self.authenticate(api_url, auth_body, attribute)
                headers = {'Authorization': f'Bearer {token}'}

            try:
                response_data = requests.get(api_url, headers=headers, params={})
                response_data.raise_for_status()
                response_data = response_data.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"API request: {e}")
                return {}, None, ""

            target_data = []
            for key in array_path.split('.'):
                target_data = target_data.get(key, [])
            array_data = target_data if isinstance(target_data, list) else []
            results.append([array_data, len(array_data), f'Bearer {token}'])
            pbar.update_absolute(idx)
        return [list(a) for a in zip(*results)]
'''
