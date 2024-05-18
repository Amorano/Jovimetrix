"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

import math
from typing import Any, Tuple

import numpy as np

from comfy.utils import ProgressBar

from Jovimetrix import comfy_message, parse_reset, JOVBaseNode, WILDCARD, \
    JOV_WEB_RES_ROOT
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.anim import EnumWave, wave_op
from Jovimetrix.sup.util import EnumConvertType, parse_param, zip_longest_fill

# =============================================================================

JOV_CATEGORY = "ANIMATE"

# =============================================================================

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) ⏱"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", WILDCARD)
    RETURN_NAMES = (Lexicon.VALUE, Lexicon.LINEAR, Lexicon.FPS, Lexicon.ANY)

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

    def run(self, ident, **kw) -> Tuple[int, float, float, Any]:
        passthru = parse_param(kw, Lexicon.ANY, EnumConvertType.ANY, None)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0)[0]
        self.__frame = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, self.__frame)[0]
        if loop > 0:
            self.__frame = min(loop, self.__frame)
        self.__frame = max(0, self.__frame)
        hold = parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False)[0]
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1)[0]
        bpm = parse_param(kw, Lexicon.BPM, EnumConvertType.INT, 120, 1)[0]
        divisor = parse_param(kw, Lexicon.NOTE, EnumConvertType.INT, 4, 1)[0]
        beat = int(fps) * 60 / max(1, int(bpm))
        beat = beat / divisor
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 1, 1)[0]
        step = 1. / max(1, int(fps))
        reset = parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]
        if parse_reset(ident) > 0 or reset:
            self.__frame = 0
            self.__fixed_step = 0
        trigger = None
        results = {
            'frame': [],
            'lin': [],
            'fixed': [],
            'trigger': [],
        }
        results = []
        pbar = ProgressBar(batch)
        for idx in range(batch):
            if passthru is not None:
                trigger = passthru if trigger else None
            lin = self.__frame if loop == 0 else self.__frame / loop
            #results['frame'].append(self.__frame)
            #results['lin'].append(lin)
            #results['fixed'].append(self.__fixed_step)
            #results['trigger'].append(trigger)
            results.append([self.__frame, lin, self.__fixed_step, trigger])
            if not hold:
                self.__frame += 1
                self.__fixed_step += step
                if loop > 0:
                    self.__frame %= loop
                self.__fixed_step = math.fmod(self.__fixed_step, fps)
                trigger = math.fmod(self.__fixed_step, beat) == 0
            pbar.update_absolute(idx)
        if loop > 0:
            self.__frame = 0
        comfy_message(ident, "jovi-tick", {"i": self.__frame})
        #return results["frame"], results["lin"], results["fixed"], results["trigger"]
        return list(zip(*results))

class WaveGeneratorNode(JOVBaseNode):
    NAME = "WAVE GENERATOR (JOV) 🌊"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = (Lexicon.FLOAT, Lexicon.INT, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.WAVE: (EnumWave._member_names_, {"default": EnumWave.SIN.name}),
            Lexicon.FREQ: ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            Lexicon.AMP: ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            Lexicon.PHASE: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001}),
            Lexicon.OFFSET: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001}),
            Lexicon.TIME: ("FLOAT", {"default": 0, "min": 0, "step": 0.000001}),
            Lexicon.INVERT: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> Tuple[float, int]:
        op = parse_param(kw, Lexicon.WAVE, EnumConvertType.STRING, EnumWave.SIN.name, enumType=EnumWave)
        freq = parse_param(kw, Lexicon.FREQ, EnumConvertType.FLOAT, 1, 0.0001)
        amp = parse_param(kw, Lexicon.AMP, EnumConvertType.FLOAT, 1, 0.0001)
        phase = parse_param(kw, Lexicon.PHASE, EnumConvertType.FLOAT, 0)
        shift = parse_param(kw, Lexicon.OFFSET, EnumConvertType.FLOAT, 0)
        delta_time = parse_param(kw, Lexicon.TIME, EnumConvertType.FLOAT, 0, 0)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        abs = parse_param(kw, Lexicon.ABSOLUTE, EnumConvertType.BOOLEAN, False)
        results = []
        params = list(zip_longest_fill(op, freq, amp, phase, shift, delta_time, invert, abs))
        pbar = ProgressBar(len(params))
        for idx, (op, freq, amp, phase, shift, delta_time, invert, abs) in enumerate(params):
            # freq = 1. / freq
            if invert:
                amp = -amp
            val = wave_op(op, phase, freq, amp, shift, delta_time)
            if abs:
                val = np.abs(val)
            results.append([val, int(val)])
            pbar.update_absolute(idx)
        return list(zip(*results))
