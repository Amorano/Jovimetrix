"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Audio
"""

from pathlib import Path
from typing import Tuple

import torch
import librosa
import numpy as np
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import ROOT, JOVBaseNode
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import EnumConvertType, parse_param, zip_longest_fill
from Jovimetrix.sup.image import channel_solid, cv2tensor_full, EnumImageType, \
    MIN_IMAGE_SIZE
from Jovimetrix.sup.audio import load_audio, graph_sausage

# =============================================================================

JOV_CATEGORY = "AUDIO"

# =============================================================================

class LoadWaveNode(JOVBaseNode):
    NAME = "LOAD WAVE (JOV) 🎼"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = (Lexicon.WAVE,)
    DESCRIPTION = """
☣️💣☣️💣☣️💣☣️💣 THIS NODE IS A WORK IN PROGRESS ☣️💣☣️💣☣️💣☣️💣

The Load Wave node imports audio files, converting them to waveforms. Specify the file path to load the audio data.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {},
            "optional": {
                Lexicon.FILEN: ("STRING", {"default": "./res/aud/bread.wav"})
        }}
        return Lexicon._parse(d, cls)

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__cache = {}

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        filen = parse_param(kw, Lexicon.FILEN, EnumConvertType.STRING, "")
        params = list(zip_longest_fill(filen))
        waves = []
        pbar = ProgressBar(len(params))
        for idx, (filen,) in enumerate(params):
            path = Path(filen)
            if not path.is_file():
                path = Path(ROOT / filen)
                if not path.is_file():
                    logger.error(f"LoadWave :: bad file {filen}")
                    waves.append(())
                    continue
            path = str(path)
            logger.info(f"LoadWave {path}")
            data = self.__cache.get(path, None)
            if data is None:
                data, rate = load_audio(path)
                data = data.astype(np.float32) / 32767.0
                self.__cache[path] = data
            waves.append(data)
            pbar.update_absolute(idx)
        return (waves,)

class WaveGraphNode(JOVBaseNode):
    NAME = "WAVE GRAPH (JOV) ▶ ılıılı"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    DESCRIPTION = """
The Wave Graph node visualizes audio waveforms as bars. Adjust parameters like the number of bars, bar thickness, and colors.
"""

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
            }
        }
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        wave = parse_param(kw, Lexicon.WAVE, EnumConvertType.ANY, None)
        bars = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 50, 1, 8192)
        thick = parse_param(kw, Lexicon.THICK, EnumConvertType.FLOAT, 0.75, 0, 1)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), MIN_IMAGE_SIZE)
        rgb_a = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, (196, 0, 196), 0, 255)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (42, 12, 42), 0, 255)
        params = list(zip_longest_fill(wave, bars, wihi, thick, rgb_a, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (wave, bars, wihi, thick, rgb_a, matte) in enumerate(params):
            width, height = wihi
            if wave is None:
                img = channel_solid(width, height, matte, EnumImageType.BGRA)
            else:
                img = graph_sausage(wave[0], bars, width, height, thickness=thick, color_line=rgb_a, color_back=matte)
            images.append(cv2tensor_full(img))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in list(zip(*images))]

class AudioWaveFilterNode(JOVBaseNode):
    NAME = "AUDIO FILTER (JOV) 🥁"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE,)
    DESCRIPTION = """
This is the template documentation comment.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {},
            "optional": {
                Lexicon.WAVE: ("AUDIO", {"default": None, "tooltip": "Audio Wave Object"}),
                Lexicon.RATE: ("INT", {"default": 22050, "min": 6000, "max": 192000, "step": 1, "tooltip": ""}),
                Lexicon.FRAME: ("INT", {"default": 1, "min": 1, "max": 262144, "step": 1, "tooltip": ""}),
                Lexicon.FPS: ("INT", {"default": 1, "min": 1, "max": 120, "step": 1, "tooltip": ""}),
            }
        }
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor]:
        wave = parse_param(kw, Lexicon.WAVE, EnumConvertType.ANY, None)
        sample_rate = parse_param(kw, Lexicon.RATE, EnumConvertType.INT, 22050, 6000, 192000)
        frame_count = parse_param(kw, Lexicon.FRAME, EnumConvertType.INT, 1, 1, 262144)
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 1, 1, 120)
        params = list(zip_longest_fill(wave, sample_rate, frame_count, fps))
        waves = []
        pbar = ProgressBar(len(params))
        for idx, (wave, sample_rate, frame_count, fps) in enumerate(params):
            timestamps = np.arange(0, frame_count) * (1 / fps)
            samples = wave.cpu().numpy()[0, :, 0]
            tempo, beats = librosa.beat.beat_track(y=samples, sr=sample_rate, hop_length=512)
            beat_timestamps = librosa.frames_to_time(beats, sr=sample_rate, hop_length=512)
            matches = librosa.util.match_events(beat_timestamps, timestamps)
            beats = np.isin(np.arange(frame_count), matches).astype(np.float32) # type: ignore
            beats_smoothed = np.zeros_like(beats)
            k_factor = 0.8 ** (60 / fps)
            beats_smoothed[0] = beats[0]
            for i in range(1, beats.shape[0]):
                beats_smoothed[i] = max(k_factor * beats_smoothed[i - 1], beats[i])
            ret = (torch.from_numpy(beats_smoothed)[:, None, None, None].expand(-1, -1, -1, 3),)
            waves.append(ret)
            pbar.update_absolute(idx)
        return list(zip(*waves))
