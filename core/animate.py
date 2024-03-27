"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

from typing import Any

from loguru import logger
import numpy as np

from comfy.utils import ProgressBar

from Jovimetrix import JOV_WEB_RES_ROOT, comfy_message, parse_reset, JOVBaseNode, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.anim import EnumWave, wave_op
from Jovimetrix.sup.util import EnumConvertType, parse_parameter, zip_longest_fill

# =============================================================================

JOV_CATEGORY = "ANIMATE"

# =============================================================================

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) ⏱"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    INPUT_IS_LIST = False
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", WILDCARD)
    RETURN_NAMES = (Lexicon.VALUE, Lexicon.LINEAR, Lexicon.FPS, Lexicon.ANY)
    OUTPUT_IS_LIST = (True, True, True, True,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            # data to pass on a pulse of the loop
            Lexicon.ANY: (WILDCARD, {"default": None, "tooltip":"Output to send when beat (BPM setting) is hit"}),
            # forces a MOD on CYCLE
            Lexicon.VALUE: ("INT", {"min": 0, "default": 0, "step": 1, "tooltip": "the current frame number of the tick"}),
            Lexicon.LOOP: ("INT", {"min": 0, "default": 0, "step": 1, "tooltip": "number of frames before looping starts. 0 means continuous playback (no loop point)"}),
            #
            Lexicon.FPS: ("INT", {"min": 1, "default": 24, "step": 1, "tooltip": "Fixed frame step rate based on FPS (1/FPS)"}),
            Lexicon.BPM: ("FLOAT", {"min": 1, "max": 60000, "default": 120, "step": 1,
                                    "tooltip": "BPM trigger rate to send the input. If input is empty, TRUE is sent on trigger"}),
            Lexicon.NOTE: ("INT", {"default": 4, "min": 1, "max": 256, "step": 1,
                                   "tooltip":"Number of beats per measure. Quarter note is 4, Eighth is 8, 16 is 16, etc."}),
            # stick the current "count"
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            # manual total = 0
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
            # how many frames to dump....
            Lexicon.BATCH: ("INT", {"min": 1, "default": 1, "step": 1, "max": 32767, "tooltip": "Number of frames wanted"}),
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        # how many pulses we have done -- total unless reset
        self.__frame = 0
        # the current frame index based on the user FPS value
        self.__fixed_step = 0

    def run(self, ident, **kw) -> tuple[int, float, float, Any]:
        passthru = parse_parameter(Lexicon.ANY, kw, None, EnumConvertType.ANY)[0]
        # how many frames before reset to 0 -- 0 = run continuous
        loop = parse_parameter(Lexicon.LOOP, kw, 0, EnumConvertType.INT)[0]
        # current frame
        self.__frame = parse_parameter(Lexicon.VALUE, kw, self.__frame, EnumConvertType.INT)[0]
        if loop > 0:
            self.__frame = min(loop, self.__frame)
        self.__frame = max(0, self.__frame)
        hold = parse_parameter(Lexicon.WAIT, kw, False, EnumConvertType.BOOLEAN)[0]
        fps = parse_parameter(Lexicon.FPS, kw, 24, EnumConvertType.INT, clip_min=1)[0]
        bpm = parse_parameter(Lexicon.BPM, kw, 120, EnumConvertType.INT, clip_min=1)[0]
        divisor = parse_parameter(Lexicon.NOTE, kw, 4, EnumConvertType.INT, clip_min=1)[0]
        beat = 240000. / max(1, int(bpm))
        beat = round(beat / divisor)
        batch = parse_parameter(Lexicon.BATCH, kw, 1, EnumConvertType.INT, clip_min=1)[0]
        results = []
        step = 1. / max(1, int(fps))
        reset = parse_parameter(Lexicon.BATCH, kw, False, EnumConvertType.BOOLEAN)[0]
        if parse_reset(ident) > 0 or reset:
            self.__frame = 0
            self.__fixed_step = 0
        trigger = None
        pbar = ProgressBar(batch)
        for idx in range(batch):
            if passthru is not None:
                trigger = passthru if trigger else None
            lin = self.__frame if loop == 0 else self.__frame / loop
            results.append([self.__frame, lin, self.__fixed_step, trigger])
            if not hold:
                self.__frame += 1
                self.__fixed_step += step
                if loop > 0:
                    self.__frame %= loop
                self.__fixed_step %= fps
                trigger = self.__frame % beat == 0
            pbar.update_absolute(idx)
        if loop > 0:
            self.__frame = 0
        comfy_message(ident, "jovi-tick", {"i": self.__frame})
        return [list(a) for a in zip(*results)]

class WaveGeneratorNode(JOVBaseNode):
    NAME = "WAVE GENERATOR (JOV) 🌊"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = (Lexicon.FLOAT, Lexicon.INT, )
    OUTPUT_IS_LIST = (True, True,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.WAVE: (EnumWave._member_names_, {"default": EnumWave.SIN.name}),
            Lexicon.FREQ: ("FLOAT", {"default": 1, "min": 0.0, "step": 0.01}),
            Lexicon.AMP: ("FLOAT", {"default": 1, "min": 0.0, "step": 0.01}),
            Lexicon.PHASE: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001}),
            Lexicon.OFFSET: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001}),
            Lexicon.TIME: ("FLOAT", {"default": 0, "min": 0, "step": 0.000001}),
            # stick the current "count"
            Lexicon.INVERT: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[float, int]:
        op = parse_parameter(Lexicon.WAVE, kw, EnumWave.SIN.name, EnumConvertType.STRING)
        freq = parse_parameter(Lexicon.FREQ, kw, 1, EnumConvertType.FLOAT, 0)
        amp = parse_parameter(Lexicon.AMP, kw, 1, EnumConvertType.FLOAT, 0)
        phase = parse_parameter(Lexicon.PHASE, kw, 0, EnumConvertType.FLOAT)
        shift = parse_parameter(Lexicon.OFFSET, kw, 0, EnumConvertType.FLOAT)
        delta_time = parse_parameter(Lexicon.TIME, kw, 0, EnumConvertType.FLOAT, 0)
        invert = parse_parameter(Lexicon.INVERT, kw, False, EnumConvertType.BOOLEAN)
        abs = parse_parameter(Lexicon.ABSOLUTE, kw, False, EnumConvertType.BOOLEAN)
        results = []
        params = [tuple(x) for x in zip_longest_fill(op, freq, amp, phase, shift, delta_time, invert, abs)]
        pbar = ProgressBar(len(params))
        for idx, (op, freq, amp, phase, shift, delta_time, invert, abs) in enumerate(params):
            freq = 1. / freq
            if invert:
                amp = -amp

            val = wave_op(op, phase, freq, amp, shift, delta_time)
            if abs:
                val = np.abs(val)
            results.append([val, int(val)])
            pbar.update_absolute(idx)
        return [list(a) for a in zip(*results)]
