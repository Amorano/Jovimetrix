"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Audio
"""

import torch
import ffmpeg
import numpy as np
from loguru import logger

import comfy

from Jovimetrix import JOV_HELP_URL, MIN_IMAGE_SIZE, JOVBaseNode, JOVImageMultiple

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_tuple, zip_longest_fill
from Jovimetrix.sup.image import EnumImageType, channel_solid, cv2tensor_full
from Jovimetrix.sup.audio import load_audio, graph_sausage

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/AUDIO"

# =============================================================================

class LoadWaveNode(JOVBaseNode):
    NAME = "LOAD WAVE (JOV) 🎼"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Import audio waveform data"
    RETURN_TYPES = ("WAVE",)
    RETURN_NAMES = (Lexicon.WAVE,)
    OUTPUT_IS_LIST = (False,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {},
            "optional": {
                Lexicon.FILEN: ("STRING", {"default": ""})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/AUDIO#-load-wave")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__cache = {}

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        filen = kw[Lexicon.FILEN]
        params = [tuple(x) for x in zip_longest_fill(filen)]
        waves = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (filen,) in enumerate(params):
            data = self.__cache.get(filen, None)
            if data is None:
                try:
                    data, rate = load_audio(filen)
                    data = data.astype(np.float32) / 32767.0
                    self.__cache[filen] = data
                except ffmpeg._run.Error as _:
                    pass
                except Exception as e:
                    logger.error(str(e))
            waves.append(data)
            pbar.update_absolute(idx)
        return waves

class WaveGraphNode(JOVImageMultiple):
    NAME = "WAVE GRAPH (JOV) ▶ ılıılı"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Display audio waveform data as a linear bar graph"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.WAVE: ("WAVE", {"default": None, "tooltip": "Audio Wave Object"}),
            Lexicon.VALUE: ("INT", {"default": 100, "min": 32, "max": 8192, "step": 1, "tooltip": "Number of Vertical bars to try to fit within the specified Width x Height"}),
            Lexicon.THICK: ("FLOAT", {"default": 0.72, "min": 0, "max": 1, "step": 0.01, "tooltip": "The percentage of fullness for each bar; currently scaled from the left only"}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE),
                                  "step": 1, "label": [Lexicon.W, Lexicon.H], "tooltip": "Final output size of the wave bar graph"}),
            Lexicon.RGBA_A: ("VEC4", {"default": (128, 128, 0, 255), "step": 1,
                                      "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True, "tooltip": "Bar Color"}),
            Lexicon.MATTE: ("VEC4", {"default": (0, 128, 128, 255), "step": 1,
                                     "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/AUDIO#-wave-graph")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        wave = kw[Lexicon.WAVE]
        bars = kw[Lexicon.VALUE]
        thick = kw[Lexicon.THICK]
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), clip_min=1)
        rgb_a = parse_tuple(Lexicon.RGBA_A, kw, default=(128, 128, 0, 255), clip_min=0, clip_max=255)
        matte = parse_tuple(Lexicon.RGBA_B, kw, default=(0, 128, 128, 255), clip_min=0, clip_max=255)
        params = [tuple(x) for x in zip_longest_fill(wave, bars, wihi, thick, rgb_a, matte)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (wave, bars, wihi, thick, rgb_a, matte) in enumerate(params):
            width, height = wihi
            if wave is None:
                img = channel_solid(width, height, matte, EnumImageType.BGRA)
            else:
                img = graph_sausage(wave, bars, width, height, thickness=thick, color_line=rgb_a, color_back=matte)
            img = cv2tensor_full(img)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))
