"""
Jovimetrix - Utility
"""

import io
import json
from typing import Any, Tuple

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ... import \
    JOV_TYPE_IMAGE, \
    InputType, Lexicon, JOVBaseNode, \
    deep_merge, parse_reset

from ...sup.util import \
    EnumConvertType, \
    parse_dynamic, parse_param

from ...sup.image import \
    MIN_IMAGE_SIZE, \
    pil2tensor

# ==============================================================================

JOV_CATEGORY = "UTILITY"

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def decode_tensor(tensor: torch.Tensor) -> str:
    if tensor.ndim > 3:
        b, h, w, cc = tensor.shape
    elif tensor.ndim > 2:
        cc = 1
        b, h, w = tensor.shape
    else:
        b = 1
        cc = 1
        h, w = tensor.shape
    return f"{b}x{w}x{h}x{cc}"

# ==============================================================================
# === CLASS -- SUPPORT ===
# ==============================================================================

class AkashicData:
    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

# ==============================================================================
# === CLASS ===
# ==============================================================================

class AkashicNode(JOVBaseNode):
    NAME = "AKASHIC (JOV) 📓"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    SORT = 10
    DESCRIPTION = """
Visualize data. It accepts various types of data, including images, text, and other types. If no input is provided, it returns an empty result. The output consists of a dictionary containing UI-related information, such as base64-encoded images and text representations of the input data.
"""

    def run(self, **kw) -> Tuple[Any, Any]:
        kw.pop('ident', None)
        o = kw.values()
        output = {"ui": {"b64_images": [], "text": []}}
        if o is None or len(o) == 0:
            output["ui"]["result"] = (None, None, )
            return output

        def __parse(val) -> str:
            ret = ''
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
                    val = val["xyzw"]
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
                        ret = json.dumps(val, indent=3, separators=(',', ': '))
                    except Exception as e:
                        ret = str(e)
            elif isinstance(val, (tuple, set, list,)):
                if (size := len(val)) > 0:
                    if isinstance(val, (np.ndarray,)):
                        ret = str(val)
                        typ = "NUMPY ARRAY"
                    elif isinstance(val[0], (torch.Tensor,)):
                        ret = decode_tensor(val[0])
                        typ = type(val[0])
                    elif size == 1 and isinstance(val[0], (list,)) and isinstance(val[0][0], (torch.Tensor,)):
                        ret = decode_tensor(val[0][0])
                        typ = "CONDITIONING"
                    elif all(isinstance(i, (tuple, set, list)) for i in val):
                        ret = "[\n" + ",\n".join(f"  {row}" for row in val) + "\n]"
                        # ret = json.dumps(val, indent=4)
                    elif all(isinstance(i, (bool, int, float)) for i in val):
                        ret = ','.join([str(x) for x in val])
                    else:
                        ret = str(val)
            elif isinstance(val, bool):
                ret = "True" if val else "False"
            elif isinstance(val, torch.Tensor):
                ret = decode_tensor(val)
            else:
                ret = str(val)
            return json.dumps({typ: ret}, separators=(',', ': '))

        for x in o:
            data = ""
            if len(x) > 1:
                data += "::\n"
            for p in x:
                data += __parse(p) + "\n"
            output["ui"]["text"].append(data)
        return output

class GraphNode(JOVBaseNode):
    NAME = "GRAPH (JOV) 📈"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE,)
    OUTPUT_TOOLTIPS = (
        "The graphed image"
    )
    SORT = 15
    DESCRIPTION = """
Visualize a series of data points over time. It accepts a dynamic number of values to graph and display, with options to reset the graph or specify the number of values. The output is an image displaying the graph, allowing users to analyze trends and patterns.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.RESET: ("BOOLEAN", {
                    "default": False,
                    "tooltip":"Clear the graph history"}),
                Lexicon.VALUE: ("INT", {
                    "default": 60, "min": 0,
                    "tooltip":"Number of values to graph and display"}),
                Lexicon.WH: ("VEC2INT", {
                    "default": (512, 512), "mij":MIN_IMAGE_SIZE,
                    "label": [Lexicon.W, Lexicon.H],
                    "tooltip":"Width and Height of the graph output"}),
            }
        })
        return Lexicon._parse(d)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("NaN")

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
    OUTPUT_TOOLTIPS = (
        "Batch count",
        "Width",
        "Height",
        "Channels",
        "Width & Height as a VEC2",
        "Width, Height and Channels as a VEC3"
    )
    SORT = 55
    DESCRIPTION = """
Exports and Displays immediate information about images.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {
                    "default": None,
                    "tooltip":"The image to examine"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> Tuple[int, list]:
        image = kw.get(Lexicon.PIXEL_A, None)
        if image.ndim == 4:
            count, height, width, cc = image.shape
        else:
            count, height, width = image.shape
            cc = 1
        return count, width, height, cc, (width, height), (width, height, cc)
