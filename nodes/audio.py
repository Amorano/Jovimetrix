"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Audio
"""

import torch
import ffmpeg
import numpy as np
from loguru import logger

import comfy

from Jovimetrix import JOVImageBaseNode, \
    MIN_IMAGE_SIZE, IT_REQUIRED, IT_WH, IT_RGBA, IT_RGBA_B

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import deep_merge_dict, parse_tuple

from Jovimetrix.sup.image import cv2mask, cv2tensor
from Jovimetrix.sup.audio import load_audio, wave_extract, graph_sausage

class GraphWaveNode(JOVImageBaseNode):
    NAME = "GRAPH WAVE (JOV) ▶ ılıılıılılılılılı. "
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/AUDIO"
    RETURN_TYPES = ("IMAGE", "MASK", "WAVE")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.MASK, Lexicon.WAVE )
    OUTPUT_IS_LIST = (False, False, True)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.FILEN: ("STRING", {"default": ""}),
                Lexicon.AMT: ("INT", {"default": 100, "min": 32, "max": 8192, "step": 1})
            }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_RGBA, IT_RGBA_B)

    # #️⃣ 🪄
    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__filen = None
        self.__data = None

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        filen = kw.get(Lexicon.FILEN)
        bars = kw.get(Lexicon.AMT, None)
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        rgb_a = parse_tuple(Lexicon.RGBA, kw, default=(128, 128, 0, 255), clip_min=1)[0]
        rgb_b = parse_tuple(Lexicon.RGBA_B, kw, default=(0, 128, 128, 255), clip_min=1)[0]

        if self.__filen != filen:
            self.__data = None
            try:
                self.__data = load_audio(filen)
                self.__filen = filen
            except ffmpeg._run.Error as _:
                pass
            except Exception as e:
                logger.error(str(e))

        image = np.zeros((1, 1), dtype=np.int16)
        if self.__data is not None:
            image = graph_sausage(self.__data, bars, width, height, rgb_a, rgb_b)

        image = cv2tensor(image)
        mask = cv2mask(image)
        data = wave_extract(self.__data)
        return (image, mask, data,)
