"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import io
import os
import sys
import json
import glob
import random
from enum import Enum
from uuid import uuid4
from pathlib import Path
from itertools import zip_longest
from typing import Any, List, Literal, Tuple

import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib.pyplot as plt

from loguru import logger

from comfy.utils import ProgressBar
from nodes import interrupt_processing
from folder_paths import get_output_directory

from Jovimetrix import JOV_TYPE_ANY, ROOT, JOV_TYPE_IMAGE, DynamicInputType, \
    Lexicon, JOVBaseNode, deep_merge, comfy_message, parse_reset

from Jovimetrix.sup.util import EnumConvertType, decode_tensor, parse_dynamic, \
    path_next, parse_param, zip_longest_fill

from Jovimetrix.sup.image import MIN_IMAGE_SIZE, IMAGE_FORMATS, EnumInterpolation, \
    EnumScaleMode, cv2tensor, cv2tensor_full, image_convert, \
    image_matte, image_scalefit, tensor2cv, pil2tensor, image_load, \
    tensor2pil

from Jovimetrix.sup.image.misc import image_by_size

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
    CARTESIAN = 40

class ContainsAnyDict(dict):
    def __contains__(self, key) -> Literal[True]:
        return True

# =============================================================================

class AkashicData:
    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

class AkashicNode(JOVBaseNode):
    NAME = "AKASHIC (JOV) 📓"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    SORT = 10
    DESCRIPTION = """
Visualize data. It accepts various types of data, including images, text, and other types. If no input is provided, it returns an empty result. The output consists of a dictionary containing UI-related information, such as base64-encoded images and text representations of the input data.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[Any, Any]:
        kw.pop('ident', None)
        o = kw.values()
        output = {"ui": {"b64_images": [], "text": []}}
        if o is None or len(o) == 0:
            output["ui"]["result"] = (None, None, )
            return output

        def __parse(val) -> str:
            ret = val
            typ = ''.join(repr(type(val)).split("'")[1:2])
            if isinstance(val, dict):
                # mixlab layer?
                if (image := val.get('image', None)) is not None:
                    ret = image
                    if (mask := val.get('mask', None)) is not None:
                        while len(mask.shape) < len(image.shape):
                            mask = mask.unsqueeze(-1)
                        ret = torch.cat((image, mask), dim=-1)
                    if ret.ndim < 4:
                        ret = ret.unsqueeze(-1)
                    ret = decode_tensor(ret)
                    typ = "Mixlab Layer"

                # vector patch....
                elif 'xyzw' in val:
                    val = {"xyzw"[i]:x for i, x in enumerate(val["xyzw"])}
                    typ = "VECTOR"
                # latents....
                elif 'samples' in val:
                    ret = decode_tensor(val['samples'][0])
                    typ = "LATENT"
                # empty bugger
                elif len(val) == 0:
                    ret = ""
                else:
                    try:
                        ret = json.dumps(val, indent=3)
                    except Exception as e:
                        ret = str(e)

            elif isinstance(val, (tuple, set, list,)):
                ret = ''
                if (size := len(val)) > 0:
                    if type(val) == np.ndarray:
                        if len(q := q()) == 1:
                            ret += f"{q[0]}"
                        elif q > 1:
                            ret += f"{q[1]}x{q[0]}"
                        else:
                            ret += f"{q[1]}x{q[0]}x{q[2]}"
                            # typ = "NUMPY ARRAY"
                    elif isinstance(val[0], (torch.Tensor,)):
                        ret = decode_tensor(val[0])
                        typ = type(val[0])
                    elif size == 1 and isinstance(val[0], (list,)) and isinstance(val[0][0], (torch.Tensor,)):
                        ret = decode_tensor(val[0][0])
                        typ = "CONDITIONING"
                    else:
                        ret = '\n\t' + '\n\t'.join(str(v) for v in val)
            elif isinstance(val, bool):
                ret = "True" if val else "False"
            elif isinstance(val, torch.Tensor):
                ret = decode_tensor(val)
            else:
                ret = str(ret)
            return f"({ret}) [{typ}]"

        for x in o:
            output["ui"]["text"].append(__parse(x))
        return output

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
                logger.debug(b.shape)
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
                logger.debug(d.shape)
                result.append(cv2tensor(d))
            data = torch.stack([r.squeeze(0) for r in result], dim=0)
            size = data.shape[0]

        if count > 0:
            data = data[0:count]

        if len(data) == 1:
            data = data[0]

        return data, size, full_list, len(full_list)

class ExportNode(JOVBaseNode):
    NAME = "EXPORT (JOV) 📽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    SORT = 2000
    DESCRIPTION = """
Responsible for saving images or animations to disk. It supports various output formats such as GIF and GIFSKI. Users can specify the output directory, filename prefix, image quality, frame rate, and other parameters. Additionally, it allows overwriting existing files or generating unique filenames to avoid conflicts. The node outputs the saved images or animation as a tensor.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.PASS_OUT: ("STRING", {"default": get_output_directory(), "default_top":"<comfy output dir>"}),
                Lexicon.FORMAT: (FORMATS, {"default": FORMATS[0]}),
                Lexicon.PREFIX: ("STRING", {"default": "jovi"}),
                Lexicon.OVERWRITE: ("BOOLEAN", {"default": False}),
                # GIF ONLY
                Lexicon.OPTIMIZE: ("BOOLEAN", {"default": False}),
                # GIFSKI ONLY
                Lexicon.QUALITY: ("INT", {"default": 90, "mij": 1, "maj": 100}),
                Lexicon.QUALITY_M: ("INT", {"default": 100, "mij": 1, "maj": 100}),
                # GIF OR GIFSKI
                Lexicon.FPS: ("INT", {"default": 24, "mij": 1, "maj": 60}),
                # GIF OR GIFSKI
                Lexicon.LOOP: ("INT", {"default": 0, "mij": 0}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> None:
        images = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        suffix = parse_param(kw, Lexicon.PREFIX, EnumConvertType.STRING, uuid4().hex[:16])[0]
        output_dir = parse_param(kw, Lexicon.PASS_OUT, EnumConvertType.STRING, "")[0]
        format = parse_param(kw, Lexicon.FORMAT, EnumConvertType.STRING, "gif")[0]
        overwrite = parse_param(kw, Lexicon.OVERWRITE, EnumConvertType.BOOLEAN, False)[0]
        optimize = parse_param(kw, Lexicon.OPTIMIZE, EnumConvertType.BOOLEAN, False)[0]
        quality = parse_param(kw, Lexicon.QUALITY, EnumConvertType.INT, 90, 0, 100)[0]
        motion = parse_param(kw, Lexicon.QUALITY_M, EnumConvertType.INT, 100, 0, 100)[0]
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1, 60)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0, 0)[0]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def output(extension) -> Path:
            path = output_dir / f"{suffix}.{extension}"
            if not overwrite and os.path.isfile(path):
                path = str(output_dir / f"{suffix}_%s.{extension}")
                path = path_next(path)
            return path

        images = [tensor2pil(i) for i in images]
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
        return ()

class GraphNode(JOVBaseNode):
    NAME = "GRAPH (JOV) 📈"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE,)
    SORT = 15
    DESCRIPTION = """
Visualize a series of data points over time. It accepts a dynamic number of values to graph and display, with options to reset the graph or specify the number of values. The output is an image displaying the graph, allowing users to analyze trends and patterns.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.RESET: ("BOOLEAN", {"default": False}),
                Lexicon.VALUE: ("INT", {"default": 60, "mij": 0, "tooltips":"Number of values to graph and display"}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]})
            },
            "outputs": {
                0: (Lexicon.IMAGE, {"tooltips":"The graphed image"}),
            }
        })
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__history = []
        self.__fig, self.__ax = plt.subplots(figsize=(5.12, 5.12))

    def run(self, ident, **kw) -> Tuple[torch.Tensor]:
        slice = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 60)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], 1)[0]
        if parse_reset(ident) > 0 or parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]:
            self.__history = []
        longest_edge = 0
        dynamic = parse_dynamic(kw, Lexicon.UNKNOWN, EnumConvertType.FLOAT, 0)
        dynamic = [i[0] for i in dynamic]
        self.__ax.clear()
        for idx, val in enumerate(dynamic):
            if isinstance(val, (set, tuple,)):
                val = list(val)
            if not isinstance(val, (list, )):
                val = [val]
            while len(self.__history) <= idx:
                self.__history.append([])
            self.__history[idx].extend(val)
            if slice > 0:
                stride = max(0, -slice + len(self.__history[idx]) + 1)
                longest_edge = max(longest_edge, stride)
                self.__history[idx] = self.__history[idx][stride:]
            self.__ax.plot(self.__history[idx], color="rgbcymk"[idx])

        self.__history = self.__history[:slice+1]
        width, height = wihi
        width, height = (width / 100., height / 100.)
        self.__fig.set_figwidth(width)
        self.__fig.set_figheight(height)
        self.__fig.canvas.draw_idle()
        buffer = io.BytesIO()
        self.__fig.savefig(buffer, format="png")
        buffer.seek(0)
        image = Image.open(buffer)
        return (pil2tensor(image),)

class ImageInfoNode(JOVBaseNode):
    NAME = "IMAGE INFO (JOV) 📚"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "VEC2", "VEC3")
    RETURN_NAMES = (Lexicon.INT, Lexicon.W, Lexicon.H, Lexicon.C, Lexicon.WH, Lexicon.WHC)
    SORT = 55
    DESCRIPTION = """
Exports and Displays immediate information about images.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE,),
            },
            "outputs": {
                0: (Lexicon.INT, {"tooltips":"Batch count"}),
                1: (Lexicon.W,),
                2: (Lexicon.H,),
                3: (Lexicon.C, {"tooltips":"Number of image channels. 1 (Grayscale), 3 (RGB) or 4 (RGBA)"}),
                4: (Lexicon.WH,),
                5: (Lexicon.WHC,),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[int, list]:
        image = kw.get(Lexicon.PIXEL_A, None)
        if image.ndim == 4:
            count, height, width, cc = image.shape
        else:
            count, height, width = image.shape
            cc = 1
        return count, width, height, cc, (width, height), (width, height, cc)

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

class RouteNode(JOVBaseNode):
    NAME = "ROUTE (JOV) 🚌"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("BUS",) + (JOV_TYPE_ANY,) * 127
    RETURN_NAMES = (Lexicon.ROUTE,)
    SORT = 850
    DESCRIPTION = """
Routes the input data from the optional input ports to the output port, preserving the order of inputs. The `PASS_IN` optional input is directly passed through to the output, while other optional inputs are collected and returned as tuples, preserving the order of insertion.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": DynamicInputType(JOV_TYPE_ANY),
            """
            "optional": {
                Lexicon.ROUTE: ("BUS", {"default": None, "tooltips":"Pass through another route node to pre-populate the outputs."}),
            },
            """
            "outputs": {
                0: (Lexicon.ROUTE, {"tooltips":"Pass through for Route node"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[Any, ...]:
        inout = parse_param(kw, Lexicon.ROUTE, EnumConvertType.ANY, None)
        vars = kw.copy()
        vars.pop(Lexicon.ROUTE, None)
        vars.pop('ident', None)
        return inout, *vars.values(),

class SaveOutput(JOVBaseNode):
    NAME = "SAVE OUTPUT (JOV) 💾"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    SORT = 85
    DESCRIPTION = """
Save the output image along with its metadata to the specified path. Supports saving additional user metadata and prompt information.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES(True, True)
        d = deep_merge(d, {
            "optional": {
                "image": ("IMAGE",),
                "path": ("STRING", {"default": "", "dynamicPrompts":False}),
                "fname": ("STRING", {"default": "output", "dynamicPrompts":False}),
                "metadata": ("JSON", {}),
                "usermeta": ("STRING", {"multiline": True, "dynamicPrompts":False,
                                        "default": ""}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> dict[str, Any]:
        image = parse_param(kw, 'image', EnumConvertType.IMAGE, None)
        metadata = parse_param(kw, 'metadata', EnumConvertType.DICT, {})
        usermeta = parse_param(kw, 'usermeta', EnumConvertType.DICT, {})
        path = parse_param(kw, 'path', EnumConvertType.STRING, "")
        fname = parse_param(kw, 'fname', EnumConvertType.STRING, "output")
        prompt = parse_param(kw, 'prompt', EnumConvertType.STRING, "")
        pnginfo = parse_param(kw, 'extra_pnginfo', EnumConvertType.DICT, {})
        params = list(zip_longest_fill(image, path, fname, metadata, usermeta, prompt, pnginfo))
        pbar = ProgressBar(len(params))
        for idx, (image, path, fname, metadata, usermeta, prompt, pnginfo) in enumerate(params):
            if image is None:
                logger.warning("no image")
                image = torch.zeros((32, 32, 4), dtype=torch.uint8, device="cpu")
            try:
                if not isinstance(usermeta, (dict,)):
                    usermeta = json.loads(usermeta)
                metadata.update(usermeta)
            except json.decoder.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(e)
                logger.error(usermeta)
            metadata["prompt"] = prompt
            metadata["workflow"] = json.dumps(pnginfo)
            image = tensor2cv(image)
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            meta_png = PngInfo()
            for x in metadata:
                try:
                    data = json.dumps(metadata[x])
                    meta_png.add_text(x, data)
                except Exception as e:
                    logger.error(e)
                    logger.error(x)
            if path == "" or path is None:
                path = get_output_directory()
            root = Path(path)
            if not root.exists():
                root = Path(get_output_directory())
            root.mkdir(parents=True, exist_ok=True)
            fname = (root / fname).with_suffix(".png")
            logger.info(f"wrote file: {fname}")
            image.save(fname, pnginfo=meta_png)
            pbar.update_absolute(idx)
        return ()
