"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import os
import sys
import json
import glob
import random
from enum import Enum
from pathlib import Path
from itertools import zip_longest
from typing import Any, List, Literal, Tuple

import torch
import numpy as np

from loguru import logger

from comfy.utils import ProgressBar
from nodes import interrupt_processing

from Jovimetrix import JOV_TYPE_ANY, ROOT, \
    Lexicon, JOVBaseNode, deep_merge, comfy_message, parse_reset

from Jovimetrix.sup.util import EnumConvertType, parse_dynamic, parse_param

from Jovimetrix.sup.image import MIN_IMAGE_SIZE, IMAGE_FORMATS, EnumInterpolation, \
    EnumScaleMode, cv2tensor, cv2tensor_full, image_convert, \
    image_matte, image_scalefit, tensor2cv, image_load

from Jovimetrix.sup.image.misc import image_by_size

# =============================================================================

JOV_CATEGORY = "UTILITY"

class EnumBatchMode(Enum):
    MERGE = 30
    PICK = 10
    SLICE = 15
    INDEX_LIST = 20
    RANDOM = 5
    CARTESIAN = 40

class ContainsAnyDict(dict):
    def __contains__(self, key) -> Literal[True]:
        return True

# =============================================================================

class ArrayNode(JOVBaseNode):
    NAME = "ARRAY (JOV) 📚"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_ANY, "INT", JOV_TYPE_ANY, "INT")
    RETURN_NAMES = (Lexicon.ANY_OUT, Lexicon.LENGTH, Lexicon.LIST, Lexicon.LENGTH2)
    SORT = 50
    DESCRIPTION = """
Processes a batch of data based on the selected mode, such as merging, picking, slicing, random selection, or indexing. Allows for flipping the order of processed items and dividing the data into chunks.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.BATCH_MODE: (EnumBatchMode._member_names_, {"default": EnumBatchMode.MERGE.name, "tooltips":"Select a single index, specific range, custom index list or randomized"}),
                Lexicon.INDEX: ("INT", {"default": 0, "mij": 0, "tooltips":"Selected list position"}),
                Lexicon.RANGE: ("VEC3INT", {"default": (0, 0, 1), "mij": 0}),
                Lexicon.STRING: ("STRING", {"default": "", "tooltips":"Comma separated list of indicies to export"}),
                Lexicon.SEED: ("INT", {"default": 0, "mij": 0, "maj": sys.maxsize}),
                Lexicon.COUNT: ("INT", {"default": 0, "mij": 0, "maj": sys.maxsize, "tooltips":"How many items to return"}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False, "tooltips":"invert the calculated output list"}),
                Lexicon.BATCH_CHUNK: ("INT", {"default": 0, "mij": 0,}),
            },
            "outputs": {
                0: (Lexicon.ANY_OUT, {"tooltips":"Output list from selected operation"}),
                1: (Lexicon.LENGTH, {"tooltips":"Length of output list"}),
                2: (Lexicon.LIST, {"tooltips":"Full list"}),
                3: (Lexicon.LENGTH2, {"tooltips":"Length of all input elements"}),
            }
        })
        return Lexicon._parse(d, cls)

    @classmethod
    def batched(cls, iterable, chunk_size, expand:bool=False, fill:Any=None) -> list:
        if expand:
            iterator = iter(iterable)
            return zip_longest(*[iterator] * chunk_size, fillvalue=fill)
        return [iterable[i: i + chunk_size] for i in range(0, len(iterable), chunk_size)]

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__seed = None

    def run(self, **kw) -> Tuple[int, list]:
        data_list = parse_dynamic(kw, Lexicon.UNKNOWN, EnumConvertType.ANY, None)
        if data_list is None:
            logger.warn("no data for list")
            return (None, [], 0)
        data_list = [item for sublist in data_list for item in sublist]
        mode = parse_param(kw, Lexicon.BATCH_MODE, EnumConvertType.STRING, EnumBatchMode.MERGE.name)[0]
        index = parse_param(kw, Lexicon.INDEX, EnumConvertType.INT, 0, 0)[0]
        slice_range = parse_param(kw, Lexicon.RANGE, EnumConvertType.VEC3INT, [(0, 0, 1)])[0]
        indices = parse_param(kw, Lexicon.STRING, EnumConvertType.STRING, "")[0]
        seed = parse_param(kw, Lexicon.SEED, EnumConvertType.INT, 0)[0]
        count = parse_param(kw, Lexicon.COUNT, EnumConvertType.INT, 0, 0, sys.maxsize)[0]
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)[0]
        batch_chunk = parse_param(kw, Lexicon.BATCH_CHUNK, EnumConvertType.INT, 0, 0)[0]

        full_list = []
        # track latents since they need to be added back to Dict['samples']
        output_is_image = False
        output_is_latent = False
        for b in data_list:
            if isinstance(b, dict) and "samples" in b:
                # latents are batched in the x.samples key
                data = b["samples"]
                full_list.extend(data)
                output_is_latent = True
            elif isinstance(b, torch.Tensor):
                # logger.debug(b.shape)
                if b.ndim == 4:
                    full_list.extend([i for i in b])
                else:
                    full_list.append(b)
                output_is_image = True
            elif isinstance(b, (list, set, tuple,)):
                full_list.extend(b)
            elif b is not None:
                full_list.append(b)

        if len(full_list) == 0:
            logger.warning("no data for list")
            return None, 0, None, 0

        data = full_list.copy()

        if flip:
            data = data[::-1]

        mode = EnumBatchMode[mode]
        if mode == EnumBatchMode.PICK:
            index = index if index < len(data) else -1
            data = [data[index]]
        elif mode == EnumBatchMode.SLICE:
            start, end, step = slice_range
            end = len(data) if end == 0 else end
            if step == 0:
                step = 1
            elif step < 0:
                data = data[::-1]
                step = abs(step)
            data = data[start:end:step]
        elif mode == EnumBatchMode.RANDOM:
            if self.__seed is None or self.__seed != seed:
                random.seed(seed)
                self.__seed = seed
            if count == 0:
                count = len(data)
            data = random.sample(data, k=count)
        elif mode == EnumBatchMode.INDEX_LIST:
            junk = []
            for x in indices.split(','):
                if '-' in x:
                    x = x.split('-')
                    a = int(x[0])
                    b = int(x[1])
                    if a > b:
                        junk = list(range(a, b-1, -1))
                    else:
                        junk = list(range(a, b + 1))
                else:
                    junk = [int(x)]
            data = [data[i:j+1] for i, j in zip(junk, junk)]

        elif mode == EnumBatchMode.CARTESIAN:
            logger.warning("NOT IMPLEMENTED - CARTESIAN")

        if len(data) == 0:
            logger.warning("no data for list")
            return None, 0, None, 0

        if batch_chunk > 0:
            data = self.batched(data, batch_chunk)

        size = len(data)
        if output_is_image:
            _, w, h = image_by_size(data)
            result = []
            for d in data:
                d = tensor2cv(d)
                d = image_convert(d, 4)
                d = image_matte(d, (0,0,0,0), w, h)
                # logger.debug(d.shape)
                result.append(cv2tensor(d))
            data = torch.stack([r.squeeze(0) for r in result], dim=0)
            size = data.shape[0]

        if count > 0:
            data = data[0:count]

        if len(data) == 1:
            data = data[0]

        return data, size, full_list, len(full_list)

class QueueBaseNode(JOVBaseNode):
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_ANY, JOV_TYPE_ANY, "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = (Lexicon.ANY_OUT, Lexicon.QUEUE, Lexicon.CURRENT, Lexicon.INDEX, Lexicon.TOTAL, Lexicon.TRIGGER, )
    VIDEO_FORMATS = IMAGE_FORMATS + ['.wav', '.mp3', '.webm', '.mp4', '.avi', '.wmv', '.mkv', '.mov', '.mxf']

    @classmethod
    def IS_CHANGED(cls, *arg, **kw) -> float:
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.QUEUE: ("STRING", {"multiline": True, "default": "./res/img/test-a.png"}),
                Lexicon.RECURSE: ("BOOLEAN", {"default": False}),
                Lexicon.BATCH: ("BOOLEAN", {"default": False, "tooltips":"Load all items, if they are loadable items, i.e. batch load images from the Queue's list. This can consume a lot of memory depending on the list size and each item size."}),
                Lexicon.VALUE: ("INT", {"mij": 0, "default": 0, "tooltips": "The current index for the current queue item"}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False, "tooltips":"Hold the item at the current queue index"}),
                Lexicon.STOP: ("BOOLEAN", {"default": False, "tooltips":"When the Queue is out of items, send a `HALT` to ComfyUI."}),
                Lexicon.LOOP: ("BOOLEAN", {"default": True, "tooltips":"If the queue should loop around the end when reached. If `False`, at the end of the Queue, if there are more iterations, it will just send the previous image."}),
                Lexicon.RESET: ("BOOLEAN", {"default": False, "tooltips":"Reset the queue back to index 1"}),
            }
        })
        return Lexicon._parse(d, cls)

    def __init__(self) -> None:
        self.__index = 0
        self.__q = None
        self.__index_last = None
        self.__len = 0
        self.__current = None
        self.__previous = None
        self.__ident = None
        self.__last_q_value = {}

    # consume the list into iterable items to load/process
    def __parseQ(self, data: Any, recurse: bool=False) -> List[str]:
        entries = []
        for line in data.strip().split('\n'):
            if len(line) == 0:
                continue

            # <directory>;png,gif,jpg
            parts = [part.strip() for part in line.split(';')]
            data = [parts[0]]
            path = Path(parts[0])
            path2 = Path(ROOT / parts[0])
            if path.exists() or path2.exists():
                philter = parts[1].split(',') if len(parts) > 1 and isinstance(parts[1], str) else self.VIDEO_FORMATS
                path = path if path.exists() else path2

                file_names = [str(path.resolve())]
                if path.is_dir():
                    if recurse:
                        file_names = [str(file.resolve()) for file in path.rglob('*') if file.is_file()]
                    else:
                        file_names = [str(file.resolve()) for file in path.iterdir() if file.is_file()]
                new_data = [fname for fname in file_names if any(fname.endswith(pat) for pat in philter)]

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
                data = [x.replace('\\', '/') for x in results]

            if len(data):
                ret = []
                for x in data:
                    try: ret.append(float(x))
                    except: ret.append(x)
                entries.extend(ret)
        return entries

    # turn Q element into actual hard type
    def process(self, q_data: Any) -> Tuple[torch.Tensor, torch.Tensor] | str | dict:
        # single Q cache to skip loading single entries over and over
        # @TODO: MRU cache strategy
        if (val := self.__last_q_value.get(q_data, None)) is not None:
            return val
        if isinstance(q_data, (str,)):
            if not os.path.isfile(q_data):
                return q_data
            _, ext = os.path.splitext(q_data)
            if ext in self.VIDEO_FORMATS:
                data = image_load(q_data)[0]
                self.__last_q_value[q_data] = data
            elif ext == '.json':
                with open(q_data, 'r', encoding='utf-8') as f:
                    self.__last_q_value[q_data] = json.load(f)
        return self.__last_q_value.get(q_data, q_data)

    def run(self, ident, **kw) -> Tuple[Any, List[str], str, int, int]:

        self.__ident = ident
        # should work headless as well
        if parse_reset(ident) > 0 or parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]:
            self.__q = None
            self.__index = 0

        if (new_val := parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, self.__index)[0]) > 0:
            self.__index = new_val - 1

        if self.__q is None:
            # process Q into ...
            # check if folder first, file, then string.
            # entry is: data, <filter if folder:*.png,*.jpg>, <repeats:1+>
            recurse = parse_param(kw, Lexicon.RECURSE, EnumConvertType.BOOLEAN, False)[0]
            q = parse_param(kw, Lexicon.QUEUE, EnumConvertType.STRING, "")[0]
            self.__q = self.__parseQ(q, recurse)
            self.__len = len(self.__q)
            self.__index_last = 0
            self.__previous = self.__q[0] if len(self.__q) else None
            if self.__previous:
                self.__previous = self.process(self.__previous)

        # make sure we have more to process if are a single fire queue
        stop = parse_param(kw, Lexicon.STOP, EnumConvertType.BOOLEAN, False)[0]
        if stop and self.__index >= self.__len:
            comfy_message(ident, "jovi-queue-done", self.status)
            interrupt_processing()
            return self.__previous, self.__q, self.__current, self.__index_last+1, self.__len

        if (wait := parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False))[0] == True:
            self.__index = self.__index_last

        # otherwise loop around the end
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.BOOLEAN, False)[0]
        if loop == True:
            self.__index %= self.__len
        else:
            self.__index = min(self.__index, self.__len-1)

        self.__current = self.__q[self.__index]
        data = self.__previous
        self.__index_last = self.__index
        info = f"QUEUE #{ident} [{self.__current}] ({self.__index})"
        batched = False
        if (batched := parse_param(kw, Lexicon.BATCH, EnumConvertType.BOOLEAN, False)[0]) == True:
            data = []
            mw, mh, mc = 0, 0, 0
            pbar = ProgressBar(self.__len)
            for idx in range(self.__len):
                ret = self.process(self.__q[idx])
                if isinstance(ret, (np.ndarray,)):
                    h, w, c = ret.shape
                    mw, mh, mc = max(mw, w), max(mh, h), max(mc, c)
                data.append(ret)
                pbar.update_absolute(idx)

            if mw != 0 or mh != 0 or mc != 0:
                ret = []
                pbar = ProgressBar(self.__len)
                for idx, d in enumerate(data):
                    d = image_convert(d, mc)
                    d = image_matte(d, (0,0,0,0), width=mw, height=mh)
                    # d = cv2tensor(d)
                    ret.append(d)
                    pbar.update_absolute(idx)
                # data = torch.cat(ret, dim=0)
                data = ret
        elif wait == True:
            info += f" PAUSED"
        else:
            data = self.process(self.__q[self.__index])
            self.__index += 1

        self.__previous = data
        comfy_message(ident, "jovi-queue-ping", self.status)
        if stop and batched:
            interrupt_processing()
        return data, self.__q, self.__current, self.__index_last+1, self.__len

    @property
    def status(self) -> dict[str, Any]:
        return {
            "id": self.__ident,
            "c": self.__current,
            "i": self.__index_last,
            "s": self.__len,
            "l": self.__q
        }

class QueueNode(QueueBaseNode):
    NAME = "QUEUE (JOV) 🗃"
    SORT = 450
    DESCRIPTION = """
Manage a queue of items, such as file paths or data. Supports various formats including images, videos, text files, and JSON files. You can specify the current index for the queue item, enable pausing the queue, or reset it back to the first index. The node outputs the current item in the queue, the entire queue, the current index, and the total number of items in the queue.
"""
    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "outputs": {
                0: (Lexicon.ANY_OUT, {"tooltips":"Current item selected from the Queue list"}),
                1: (Lexicon.QUEUE, {"tooltips":"The entire Queue list"}),
                2: (Lexicon.CURRENT, {"tooltips":"Current item selected from the Queue list as a string"}),
                3: (Lexicon.INDEX, {"tooltips":"Current index for the selected item in the Queue list"}),
                4: (Lexicon.TOTAL, {"tooltips":"Total items in the current Queue List"}),
                5: (Lexicon.TRIGGER, {"tooltips":"Send a True signal when the queue end index is reached"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, ident, **kw) -> Tuple[Any, List[str], str, int, int]:
        data, aa, ba, ca, da = super().run(ident, **kw)
        if isinstance(data, (list, )):
            if isinstance(data[0], (np.ndarray,)):
                data = [cv2tensor(d) for d in data]
                data = torch.cat(data, dim=0)
        elif isinstance(data, (np.ndarray,)):
            data = cv2tensor(data)
        return data, aa, ba, ca, da

class QueueTooNode(QueueBaseNode):
    NAME = "QUEUE TOO (JOV) 🗃"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK, Lexicon.CURRENT, Lexicon.INDEX, Lexicon.TOTAL, Lexicon.TRIGGER, )
    SORT = 500
    DESCRIPTION = """
Manage a queue of specific items: media files. Supports various image and video formats. You can specify the current index for the queue item, enable pausing the queue, or reset it back to the first index. The node outputs the current item in the queue, the entire queue, the current index, and the total number of items in the queue.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = {
            "optional": {
                Lexicon.QUEUE: ("STRING", {"multiline": True, "default": "./res/img/test-a.png"}),
                Lexicon.RECURSE: ("BOOLEAN", {"default": False}),
                Lexicon.BATCH: ("BOOLEAN", {"default": False, "tooltips":"Load all items, if they are loadable items, i.e. batch load images from the Queue's list"}),
                Lexicon.VALUE: ("INT", {"mij": 0, "default": 0, "tooltips": "The current index for the current queue item"}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False, "tooltips":"Hold the item at the current queue index"}),
                Lexicon.STOP: ("BOOLEAN", {"default": False, "tooltips":"When the Queue is out of items, send a `HALT` to ComfyUI."}),
                Lexicon.LOOP: ("BOOLEAN", {"default": True, "tooltips":"If the queue should loop around the end when reached. If `False`, at the end of the Queue, if there are more iterations, it will just send the previous image."}),
                Lexicon.RESET: ("BOOLEAN", {"default": False, "tooltips":"Reset the queue back to index 1"}),
                #
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
            },
            "outputs": {
                0: ("IMAGE", {"tooltips":"Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output."}),
                1: ("IMAGE", {"tooltips":"Three channel [RGB] image. There will be no alpha."}),
                2: ("MASK", {"tooltips":"Single channel mask output."}),
                3: (Lexicon.CURRENT, {"tooltips":"Current item selected from the Queue list as a string"}),
                4: (Lexicon.INDEX, {"tooltips":"Current index for the selected item in the Queue list"}),
                5: (Lexicon.TOTAL, {"tooltips":"Total items in the current Queue List"}),
                6: (Lexicon.TRIGGER, {"tooltips":"Send a True signal when the queue end index is reached"}),
            },
            "hidden": d.get("hidden", {}),
        }
        return Lexicon._parse(d, cls)

    def run(self, ident, **kw) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, int]:
        data, _, current, index, total = super().run(ident, **kw)
        if not isinstance(data, (list, )):
            data = [data]

        mode = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, EnumScaleMode.MATTE.name)[0]
        mode = EnumScaleMode[mode]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)[0]
        w, h = wihi
        sample = parse_param(kw, Lexicon.SAMPLE, EnumConvertType.STRING, EnumInterpolation.LANCZOS4.name)[0]
        sample = EnumInterpolation[sample]
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)[0]
        images = []
        pbar = ProgressBar(len(data))
        for idx, image in enumerate(data):
            if mode != EnumScaleMode.MATTE:
                image = tensor2cv(image)
                image = image_scalefit(image, w, h, mode, sample)
            images.append(cv2tensor_full(image, matte))
            pbar.update_absolute(idx)
        images = [torch.cat(i, dim=0) for i in zip(*images)]
        return *images, current, index, total
