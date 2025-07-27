""" Jovimetrix - Utility """

import os
import sys
import json
import glob
import random
from enum import Enum
from pathlib import Path
from itertools import zip_longest
from typing import Any

import torch
import numpy as np

from comfy.utils import ProgressBar
from nodes import interrupt_processing

from cozy_comfyui import \
    logger, \
    IMAGE_SIZE_MIN, \
    InputType, EnumConvertType, TensorType, \
    deep_merge, parse_dynamic, parse_param

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_ANY, \
    CozyBaseNode

from cozy_comfyui.image import \
    IMAGE_FORMATS

from cozy_comfyui.image.compose import \
    EnumScaleMode, EnumInterpolation, \
    image_matte, image_scalefit

from cozy_comfyui.image.convert import \
    image_convert, cv_to_tensor, cv_to_tensor_full, tensor_to_cv

from cozy_comfyui.image.misc import \
    image_by_size

from cozy_comfyui.image.io import \
    image_load

from cozy_comfyui.api import \
    parse_reset, comfy_api_post

from ... import \
    ROOT

JOV_CATEGORY = "UTILITY/BATCH"

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumBatchMode(Enum):
    MERGE = 30
    PICK = 10
    SLICE = 15
    INDEX_LIST = 20
    RANDOM = 5

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ArrayNode(CozyBaseNode):
    NAME = "ARRAY (JOV) 📚"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, "INT",)
    RETURN_NAMES = ("ARRAY", "LENGTH",)
    OUTPUT_IS_LIST = (True, True,)
    OUTPUT_TOOLTIPS = (
        "Output list from selected operation",
        "Length of output list",
        "Full input list",
        "Length of all input elements",
    )
    DESCRIPTION = """
Processes a batch of data based on the selected mode. Merge, pick, slice, random select, or index items. Can also reverse the order of items.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.MODE: (EnumBatchMode._member_names_, {
                    "default": EnumBatchMode.MERGE.name,
                    "tooltip": "Select a single index, specific range, custom index list or randomized"}),
                Lexicon.RANGE: ("VEC3", {
                    "default": (0, 0, 1), "mij": 0, "int": True,
                    "tooltip": "The start, end and step for the range"}),
                Lexicon.INDEX: ("STRING", {
                    "default": "",
                    "tooltip": "Comma separated list of indicies to export"}),
                Lexicon.COUNT: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize,
                    "tooltip": "How many items to return"}),
                Lexicon.REVERSE: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the calculated output list"}),
                Lexicon.SEED: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize}),
            }
        })
        return Lexicon._parse(d)

    @classmethod
    def batched(cls, iterable, chunk_size, expand:bool=False, fill:Any=None) -> list[Any]:
        if expand:
            iterator = iter(iterable)
            return zip_longest(*[iterator] * chunk_size, fillvalue=fill)
        return [iterable[i: i + chunk_size] for i in range(0, len(iterable), chunk_size)]

    def run(self, **kw) -> tuple[int, list]:
        data_list = parse_dynamic(kw, Lexicon.DYNAMIC, EnumConvertType.ANY, None)
        mode = parse_param(kw, Lexicon.MODE, EnumBatchMode, EnumBatchMode.MERGE.name)[0]
        slice_range = parse_param(kw, Lexicon.RANGE, EnumConvertType.VEC3INT, (0, 0, 1))[0]
        index = parse_param(kw, Lexicon.INDEX, EnumConvertType.STRING, "")[0]
        count = parse_param(kw, Lexicon.COUNT, EnumConvertType.INT, 0, 0)[0]
        reverse = parse_param(kw, Lexicon.REVERSE, EnumConvertType.BOOLEAN, False)[0]
        seed = parse_param(kw, Lexicon.SEED, EnumConvertType.INT, 0, 0)[0]

        data = []
        # track latents since they need to be added back to Dict['samples']
        output_type = None
        for b in data_list:
            if isinstance(b, dict) and "samples" in b:
                # latents are batched in the x.samples key
                if output_type and output_type != EnumConvertType.LATENT:
                    raise Exception(f"Cannot mix input types {output_type} vs {EnumConvertType.LATENT}")
                data.extend(b["samples"])
                output_type = EnumConvertType.LATENT

            elif isinstance(b, TensorType):
                if output_type and output_type not in (EnumConvertType.IMAGE, EnumConvertType.MASK):
                    raise Exception(f"Cannot mix input types {output_type} vs {EnumConvertType.IMAGE}")

                if b.ndim == 4:
                    b = [i for i in b]
                else:
                    b = [b]

                for x in b:
                    if x.ndim == 2:
                        x = x.unsqueeze(-1)
                    data.append(x)

                output_type = EnumConvertType.IMAGE

            elif b is not None:
                idx_type = type(b)
                if output_type and output_type != idx_type:
                    raise Exception(f"Cannot mix input types {output_type} vs {idx_type}")
                data.append(b)

        if len(data) == 0:
            logger.warning("no data for list")
            return [], [0], [], [0]

        if mode == EnumBatchMode.PICK:
            start, end, step = slice_range
            start = start if start < len(data) else -1
            data = [data[start]]
        elif mode == EnumBatchMode.SLICE:
            start, end, step = slice_range
            start = abs(start)
            end = len(data) if end == 0 else abs(end+1)
            if step == 0:
                step = 1
            elif step < 0:
                data = data[::-1]
                step = abs(step)
            data = data[start:end:step]
        elif mode == EnumBatchMode.RANDOM:
            random.seed(seed)
            if count == 0:
                count = len(data)
            else:
                count = max(1, min(len(data), count))
            data = random.sample(data, k=count)
        elif mode == EnumBatchMode.INDEX_LIST:
            junk = []
            for x in index.split(','):
                if '-' in x:
                    x = x.split('-')
                    for idx, v in enumerate(x):
                        try:
                            x[idx] = max(0, min(len(data)-1, int(v)))
                        except ValueError as e:
                            logger.error(e)
                            x[idx] = 0

                    if x[0] > x[1]:
                        tmp = list(range(x[0], x[1]-1, -1))
                    else:
                        tmp = list(range(x[0], x[1]+1))
                    junk.extend(tmp)
                else:
                    idx = max(0, min(len(data)-1, int(x)))
                    junk.append(idx)
            if len(junk) > 0:
                data = [data[i] for i in junk]

        if len(data) == 0:
            logger.warning("no data for list")
            return [], [0], [], [0]

        # reverse before?
        if reverse:
            data.reverse()

        # cut the list down first
        if count > 0:
            data = data[0:count]

        size = len(data)
        if output_type == EnumConvertType.IMAGE:
            _, w, h = image_by_size(data)
            result = []
            for d in data:
                w2, h2, cc = d.shape
                if w != w2 or h != h2 or cc != 4:
                    d = tensor_to_cv(d)
                    d = image_convert(d, 4)
                    d = image_matte(d, (0,0,0,0), w, h)
                    d = cv_to_tensor(d)
                d = d.unsqueeze(0)
                result.append(d)

            size = len(result)
            data = torch.stack(result)
        else:
            data = [data]

        return (data, [size],)

class BatchToList(CozyBaseNode):
    NAME = "BATCH TO LIST (JOV)"
    NAME_PRETTY = "BATCH TO LIST (JOV)"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, )
    RETURN_NAMES = ("LIST", )
    DESCRIPTION = """
Convert a batch of values into a pure python list of values.
"""
    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        return deep_merge(d, {
            "optional": {
                Lexicon.BATCH: (COZY_TYPE_ANY, {}),
            }
        })

    def run(self, **kw) -> tuple[list[Any]]:
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.LIST, [])
        batch = [f[0] for f in batch]
        return (batch,)

class QueueBaseNode(CozyBaseNode):
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, COZY_TYPE_ANY, "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("❔", "QUEUE", "CURRENT", "INDEX", "TOTAL", "TRIGGER", )
    #OUTPUT_IS_LIST = (True, True, True, True, True, True,)
    VIDEO_FORMATS = ['.wav', '.mp3', '.webm', '.mp4', '.avi', '.wmv', '.mkv', '.mov', '.mxf']

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float('nan')

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.QUEUE: ("STRING", {
                    "default": "./res/img/test-a.png", "multiline": True,
                    "tooltip": "Current items to process during Queue iteration"}),
                Lexicon.RECURSE: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Recurse through all subdirectories found"}),
                Lexicon.BATCH: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load all items, if they are loadable items, i.e. batch load images from the Queue's list"}),
                Lexicon.SELECT: ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "The index to use for the current queue item. 0 will move to the next item each queue run"}),
                Lexicon.HOLD: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Hold the item at the current queue index"}),
                Lexicon.STOP: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When the Queue is out of items, send a `HALT` to ComfyUI"}),
                Lexicon.LOOP: ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If the queue should loop. If `False` and if there are more iterations, will send the previous image"}),
                Lexicon.RESET: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reset the queue back to index 1"}),
            }
        })
        return Lexicon._parse(d)

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
    def __parseQ(self, data: Any, recurse: bool=False) -> list[str]:
        entries = []
        for line in data.strip().split('\n'):
            if len(line) == 0:
                continue

            data = [line]
            if not line.lower().startswith("http"):
                # <directory>;*.png;*.gif;*.jpg
                base_path_str, tail = os.path.split(line)
                filters = [p.strip() for p in tail.split(';')]

                base_path = Path(base_path_str)
                if base_path.is_absolute():
                    search_dir = base_path if base_path.is_dir() else base_path.parent
                else:
                    search_dir = (ROOT / base_path).resolve()

                # Check if the base directory exists
                if search_dir.exists():
                    if search_dir.is_dir():
                        new_data = []
                        filters = filters if len(filters) > 0 and isinstance(filters[0], str) else IMAGE_FORMATS
                        for pattern in filters:
                            found = glob.glob(str(search_dir / pattern), recursive=recurse)
                            new_data.extend([str(Path(f).resolve()) for f in found if Path(f).is_file()])
                        if len(new_data):
                            data = new_data
                    elif search_dir.is_file():
                        path = str(search_dir.resolve())
                        if path.lower().endswith('.txt'):
                            with open(path, 'r', encoding='utf-8') as f:
                                data = f.read().split('\n')
                        else:
                            data = [path]
                elif len(results := glob.glob(str(search_dir))) > 0:
                    data = [x.replace('\\', '/') for x in results]

            if len(data):
                ret = []
                for x in data:
                    try: ret.append(float(x))
                    except: ret.append(x)
                entries.extend(ret)
        return entries

    # turn Q element into actual hard type
    def process(self, q_data: Any) -> TensorType | str | dict:
        # single Q cache to skip loading single entries over and over
        # @TODO: MRU cache strategy
        if (val := self.__last_q_value.get(q_data, None)) is not None:
            return val
        if isinstance(q_data, (str,)):
            _, ext = os.path.splitext(q_data)
            if ext in IMAGE_FORMATS:
                data = image_load(q_data)[0]
                self.__last_q_value[q_data] = data
            #elif ext in self.VIDEO_FORMATS:
            #    data = load_file(q_data)
            #    self.__last_q_value[q_data] = data
            elif ext == '.json':
                with open(q_data, 'r', encoding='utf-8') as f:
                    self.__last_q_value[q_data] = json.load(f)
        return self.__last_q_value.get(q_data, q_data)

    def run(self, ident, **kw) -> tuple[Any, list[str], str, int, int]:

        self.__ident = ident
        # should work headless as well

        if (new_val := parse_param(kw, Lexicon.SELECT, EnumConvertType.INT, 0)[0]) > 0:
            self.__index = new_val - 1

        reset = parse_reset(ident) > 0
        if reset or parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]:
            self.__q = None
            self.__index = 0

        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)[0]
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)[0]
        w, h = wihi
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)[0]

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
            comfy_api_post("jovi-queue-done", ident, self.status)
            interrupt_processing()
            return self.__previous, self.__q, self.__current, self.__index_last+1, self.__len

        if (wait := parse_param(kw, Lexicon.HOLD, EnumConvertType.BOOLEAN, False))[0] == True:
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
            for idx in range(self.__len):
                ret = self.process(self.__q[idx])
                if isinstance(ret, (np.ndarray,)):
                    h2, w2, c = ret.shape
                    mw, mh, mc = max(mw, w2), max(mh, h2), max(mc, c)
                data.append(ret)

            if mw != 0 or mh != 0 or mc != 0:
                ret = []
                # matte = [matte[0], matte[1], matte[2], 0]
                pbar = ProgressBar(self.__len)
                for idx, d in enumerate(data):
                    d = image_convert(d, mc)
                    if mode != EnumScaleMode.MATTE:
                        d = image_scalefit(d, w, h, mode, sample, matte)
                        d = image_scalefit(d, w, h, EnumScaleMode.RESIZE_MATTE, sample, matte)
                    else:
                        d = image_matte(d, matte, mw, mh)
                    ret.append(cv_to_tensor(d))
                    pbar.update_absolute(idx)
                data = torch.stack(ret)
        elif wait == True:
            info += f" PAUSED"
        else:
            data = self.process(self.__q[self.__index])
            if isinstance(data, (np.ndarray,)):
                if mode != EnumScaleMode.MATTE:
                    data = image_scalefit(data, w, h, mode, sample)
                data = cv_to_tensor(data).unsqueeze(0)
            self.__index += 1

        self.__previous = data
        comfy_api_post("jovi-queue-ping", ident, self.status)
        if stop and batched:
            interrupt_processing()
        return data, self.__q, self.__current, self.__index, self.__len, self.__index == self.__len or batched

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
    OUTPUT_TOOLTIPS = (
        "Current item selected from the Queue list",
        "The entire Queue list",
        "Current item selected from the Queue list as a string",
        "Current index for the selected item in the Queue list",
        "Total items in the current Queue List",
        "Send a True signal when the queue end index is reached"
    )
    DESCRIPTION = """
Manage a queue of items, such as file paths or data. Supports various formats including images, videos, text files, and JSON files. You can specify the current index for the queue item, enable pausing the queue, or reset it back to the first index. The node outputs the current item in the queue, the entire queue, the current index, and the total number of items in the queue.
"""

class QueueTooNode(QueueBaseNode):
    NAME = "QUEUE TOO (JOV) 🗃"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("RGBA", "RGB", "MASK", "CURRENT", "INDEX", "TOTAL", "TRIGGER", )
    #OUTPUT_IS_LIST = (False, False, False, True, True, True, True,)
    OUTPUT_TOOLTIPS = (
        "Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output",
        "Three channel [RGB] image. There will be no alpha",
        "Single channel mask output",
        "Current item selected from the Queue list as a string",
        "Current index for the selected item in the Queue list",
        "Total items in the current Queue List",
        "Send a True signal when the queue end index is reached"
    )
    DESCRIPTION = """
Manage a queue of specific items: media files. Supports various image and video formats. You can specify the current index for the queue item, enable pausing the queue, or reset it back to the first index. The node outputs the current item in the queue, the entire queue, the current index, and the total number of items in the queue.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"],}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
            },
            "hidden": d.get("hidden", {})
        })
        return Lexicon._parse(d)

    def run(self, ident, **kw) -> tuple[TensorType, TensorType, TensorType, str, int, int, bool]:
        data, _, current, index, total, trigger = super().run(ident, **kw)
        if not isinstance(data, (TensorType, )):
            data = [None, None, None]
        else:
            matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)[0]
            data = [tensor_to_cv(d) for d in data]
            data = [cv_to_tensor_full(d, matte) for d in data]
            data = [torch.stack(d) for d in zip(*data)]
        return *data, current, index, total, trigger
