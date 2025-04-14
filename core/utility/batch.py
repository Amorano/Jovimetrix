""" Jovimetrix - Utility """

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

from comfy.utils import ProgressBar
from nodes import interrupt_processing

from cozy_comfyui import \
    logger, \
    IMAGE_SIZE_MIN, \
    InputType, EnumConvertType, TensorType, \
    deep_merge, parse_dynamic, parse_param

from cozy_comfyui.node import \
    COZY_TYPE_ANY, \
    CozyBaseNode

from cozy_comfyui.image import \
    IMAGE_FORMATS

from cozy_comfyui.image.convert import \
    image_convert, cv_to_tensor, cv_to_tensor_full, tensor_to_cv, image_matte

from cozy_comfyui.image.misc import \
    EnumInterpolation, \
    image_load

from cozy_comfyui.api import \
    parse_reset, comfy_api_post

from ... import \
    ROOT, \
    Lexicon

from ...sup.image.adjust import \
    EnumScaleMode, \
    image_scalefit

# ==============================================================================

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

# ==============================================================================

class ArrayNode(CozyBaseNode):
    NAME = "ARRAY (JOV) 📚"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    INPUT_IS_LIST = True
    RETURN_TYPES = (COZY_TYPE_ANY, "INT", COZY_TYPE_ANY, "INT", COZY_TYPE_ANY)
    RETURN_NAMES = (Lexicon.ANY_OUT, Lexicon.LENGTH, Lexicon.LIST, Lexicon.LENGTH2, Lexicon.LIST)
    OUTPUT_IS_LIST = (False, False, False, False, True)
    OUTPUT_TOOLTIPS = (
        "Output list from selected operation",
        "Length of output list",
        "Full list",
        "Length of all input elements",
        "The elements as a COMFYUI list output"
    )
    SORT = 50
    DESCRIPTION = """
Processes a batch of data based on the selected mode, such as merging, picking, slicing, random selection, or indexing. Allows for flipping the order of processed items and dividing the data into chunks.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.BATCH_MODE: (EnumBatchMode._member_names_, {
                    "default": EnumBatchMode.MERGE.name,
                    "tooltip":"Select a single index, specific range, custom index list or randomized"}),
                Lexicon.INDEX: ("INT", {
                    "default": 0, "min": 0,
                    "tooltip":"Selected list position"}),
                Lexicon.RANGE: ("VEC3INT", {
                    "default": (0, 0, 1), "mij": 0,
                    "tooltip":"The start, end and step for the range"}),
                Lexicon.STRING: ("STRING", {
                    "default": "",
                    "tooltip":"Comma separated list of indicies to export"}),
                Lexicon.SEED: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize,
                    "tooltip":"Random seed value"}),
                Lexicon.COUNT: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize,
                    "tooltip":"How many items to return"}),
                Lexicon.FLIP: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"reverse the calculated output list"}),
                Lexicon.BATCH_CHUNK: ("INT", {
                    "default": 0, "min": 0,
                    "tooltip":"How many items to put inside each 'batched' output. 0 means put all items in a single batch."}),
            }
        })
        return Lexicon._parse(d)

    @classmethod
    def batched(cls, iterable, chunk_size, expand:bool=False, fill:Any=None) -> List[Any]:
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
        mode = parse_param(kw, Lexicon.BATCH_MODE, EnumBatchMode, EnumBatchMode.MERGE.name)[0]
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
            elif isinstance(b, TensorType):
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

        if flip:
            full_list.reverse()

        data = full_list.copy()

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
            # _, w, h = image_by_size(data)
            result = []
            for d in data:
                d = tensor_to_cv(d)
                d = image_convert(d, 4)
                #d = image_matte(d, (0,0,0,0), w, h)
                # logger.debug(d.shape)
                result.append(cv_to_tensor(d))

            if len(result) > 1:
                data = torch.stack(result)
            else:
                data = result[0].unsqueeze(0)
            size = data.shape[0]

        if count > 0:
            data = data[0:count]

        if not output_is_image and len(data) == 1:
            data = data[0]

        return data, size, full_list, len(full_list), data

class QueueBaseNode(CozyBaseNode):
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (COZY_TYPE_ANY, COZY_TYPE_ANY, "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = (Lexicon.ANY_OUT, Lexicon.QUEUE, Lexicon.CURRENT, Lexicon.INDEX, Lexicon.TOTAL, Lexicon.TRIGGER, )
    VIDEO_FORMATS = ['.wav', '.mp3', '.webm', '.mp4', '.avi', '.wmv', '.mkv', '.mov', '.mxf']

    @classmethod
    def IS_CHANGED(cls, *arg, **kw) -> float:
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.QUEUE: ("STRING", {
                    "default": "./res/img/test-a.png", "multiline": True}),
                Lexicon.RECURSE: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"Recurse through all subdirectories found"}),
                Lexicon.BATCH: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"Load all items, if they are loadable items, i.e. batch load images from the Queue's list."}),
                Lexicon.VALUE: ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "The current index for the current queue item"}),
                Lexicon.WAIT: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"Hold the item at the current queue index"}),
                Lexicon.STOP: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"When the Queue is out of items, send a `HALT` to ComfyUI."}),
                Lexicon.LOOP: ("BOOLEAN", {
                    "default": True,
                    "tooltip":"If the queue should loop. If `False` and if there are more iterations, will send the previous image."}),
                Lexicon.RESET: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"Reset the queue back to index 1"}),
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
    def __parseQ(self, data: Any, recurse: bool=False) -> List[str]:
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

    def run(self, ident, **kw) -> Tuple[Any, List[str], str, int, int]:

        self.__ident = ident
        # should work headless as well
        if parse_reset(ident) > 0 or parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]:
            self.__q = None
            self.__index = 0

        if (new_val := parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 0)[0]) > 0:
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
            comfy_api_post("jovi-queue-done", ident, self.status)
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
            for idx in range(self.__len):
                ret = self.process(self.__q[idx])
                if isinstance(ret, (np.ndarray,)):
                    h, w, c = ret.shape
                    mw, mh, mc = max(mw, w), max(mh, h), max(mc, c)
                data.append(ret)

            if mw != 0 or mh != 0 or mc != 0:
                ret = []
                mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)[0]
                sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)[0]
                wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], IMAGE_SIZE_MIN)[0]
                w2, h2 = wihi
                matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)[0]
                matte = [matte[0], matte[1], matte[2], 0]
                pbar = ProgressBar(self.__len)

                for idx, d in enumerate(data):
                    d = image_convert(d, mc)
                    if mode != EnumScaleMode.MATTE:
                        d = image_scalefit(d, w2, h2, mode=mode, sample=sample)
                    else:
                        d = image_matte(d, matte, width=mw, height=mh)
                    ret.append(cv_to_tensor(d))
                    pbar.update_absolute(idx)
                data = torch.stack(ret)
        elif wait == True:
            info += f" PAUSED"
        else:
            data = self.process(self.__q[self.__index])
            if isinstance(data, (np.ndarray,)):
                data = cv_to_tensor(data).unsqueeze(0)
            self.__index += 1

        self.__previous = data
        comfy_api_post("jovi-queue-ping", ident, self.status)
        if stop and batched:
            interrupt_processing()
        return data, self.__q, self.__current, self.__index, self.__len, self.__index == self.__index_last or batched

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
    SORT = 450
    DESCRIPTION = """
Manage a queue of items, such as file paths or data. Supports various formats including images, videos, text files, and JSON files. You can specify the current index for the queue item, enable pausing the queue, or reset it back to the first index. The node outputs the current item in the queue, the entire queue, the current index, and the total number of items in the queue.
"""

class QueueTooNode(QueueBaseNode):
    NAME = "QUEUE TOO (JOV) 🗃"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK, Lexicon.CURRENT, Lexicon.INDEX, Lexicon.TOTAL, Lexicon.TRIGGER, )
    OUTPUT_TOOLTIPS = (
        "Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output",
        "Three channel [RGB] image. There will be no alpha",
        "Single channel mask output",
        "Current item selected from the Queue list as a string",
        "Current index for the selected item in the Queue list",
        "Total items in the current Queue List",
        "Send a True signal when the queue end index is reached"
    )
    SORT = 500
    DESCRIPTION = """
Manage a queue of specific items: media files. Supports various image and video formats. You can specify the current index for the queue item, enable pausing the queue, or reset it back to the first index. The node outputs the current item in the queue, the entire queue, the current index, and the total number of items in the queue.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.QUEUE: ("STRING", {
                    "default": "./res/img/test-a.png", "multiline": True,
                    "tooltip": ""}),
                Lexicon.RECURSE: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Search within sub-directories"}),
                Lexicon.BATCH: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"Load all items, if they are loadable items, i.e. batch load images from the Queue's list"}),
                Lexicon.VALUE: ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Current index for the current queue item"}),
                Lexicon.WAIT: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"Hold the item at the current queue index"}),
                Lexicon.STOP: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"When the Queue is out of items, send a `HALT` to ComfyUI."}),
                Lexicon.LOOP: ("BOOLEAN", {
                    "default": True,
                    "tooltip":"If the queue should loop. If `False` and there are more iterations, will send the previous image."}),
                Lexicon.RESET: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"Reset the queue back to index 1"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name,
                    "tooltip": "Decide whether the images should be resized to fit"}),
                Lexicon.WH: ("VEC2INT", {
                    "default": (512, 512), "mij":IMAGE_SIZE_MIN,
                    "label": [Lexicon.W, Lexicon.H],
                    "tooltip": "Width and Height"}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,
                    "tooltip": "Method for resizing images."}),
                Lexicon.MATTE: ("VEC4INT", {
                    "default": (0, 0, 0, 255), "rgb": True,
                    "tooltip": "Background color for padding"}),
            },
            "hidden": d.get("hidden", {})
        })
        return Lexicon._parse(d)

    def run(self, ident, **kw) -> Tuple[TensorType, TensorType, TensorType, str, int, int, bool]:
        data, _, current, index, total, trigger = super().run(ident, **kw)
        if not isinstance(data, (TensorType, )):
            data = [None, None, None]
        else:
            matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)[0]
            data = [tensor_to_cv(d) for d in data]
            data = [cv_to_tensor_full(d, matte) for d in data]
            data = [torch.stack(d) for d in zip(*data)]
        return *data, current, index, total, trigger
