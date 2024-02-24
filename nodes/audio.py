"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Audio
"""

import torch
import ffmpeg
import numpy as np
from loguru import logger

import comfy

from Jovimetrix import JOV_HELP_URL, JOVBaseNode, \
    MIN_IMAGE_SIZE, IT_REQUIRED, IT_WH, IT_RGBA_A, IT_RGBA_B

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import deep_merge_dict, parse_tuple

from Jovimetrix.sup.image import cv2tensor, image_mask
from Jovimetrix.sup.audio import load_audio, wave_extract, graph_sausage

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/AUDIO"

# =============================================================================

class GraphWaveNode(JOVBaseNode):
    NAME = "GRAPH WAVE (JOV) ▶ ılıılı"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Import and display audio linear waveform data."
    RETURN_TYPES = ("IMAGE", "MASK", "WAVE")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.MASK, Lexicon.WAVE )
    OUTPUT_IS_LIST = (False, False, True)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.FILEN: ("STRING", {"default": ""}),
                Lexicon.VALUE: ("INT", {"default": 100, "min": 32, "max": 8192, "step": 1})
            }}

        d = deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_RGBA_A, IT_RGBA_B)
        return Lexicon._parse(d, JOV_HELP_URL + "/AUDIO#-graph-wave")

    # #️⃣ 🪄
    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__filen = None
        self.__data = None

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        filen = kw.get(Lexicon.FILEN)
        bars = kw.get(Lexicon.VALUE, None)
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), clip_min=1)[0]
        rgb_a = parse_tuple(Lexicon.RGB_A, kw, default=(128, 128, 0, 255), clip_min=1)[0]
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

        data = wave_extract(self.__data)
        return cv2tensor(image), image_mask(image), data
