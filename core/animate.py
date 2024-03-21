"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

import math
import time
from typing import Any

from loguru import logger
import numpy as np

from comfy.utils import ProgressBar

from Jovimetrix import JOV_HELP_URL, WILDCARD, JOVBaseNode, comfy_message, parse_reset
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.anim import EnumWave, wave_op
from Jovimetrix.sup.util import zip_longest_fill

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"

# =============================================================================

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) ⏱"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Periodic pulse with total pulse count, normalized count relative to the loop setting and fixed pulse step."
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True, True, True, True,)
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
            Lexicon.VALUE: ("INT", {"min": 0, "default": 0, "step": 1, "tooltip": "current tick index"}),
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
            Lexicon.BATCH: ("INT", {"min": 1, "default": 1, "step": 1, "tooltip": "Number of frames wanted"}),
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/ANIMATE#-tick")

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        # how many pulses we have done -- total unless reset
        self.__count = 1
        # the current frame index based on the user FPS value
        self.__fixed_step = 0

    def run(self, ident, **kw) -> tuple[int, float, float, Any]:
        passthru = kw.get(Lexicon.ANY, None)
        self.__count = kw.get(Lexicon.VALUE, self.__count)
        loop = kw.get(Lexicon.LOOP, 0)
        hold = kw.get(Lexicon.WAIT, False)
        fps = kw.get(Lexicon.FPS, 24)
        bpm = kw.get(Lexicon.BPM, 120)
        divisor = kw.get(Lexicon.NOTE, 4)
        beat = 240000. / max(1, int(bpm))
        beat = round(beat / divisor)
        batch = kw.get(Lexicon.BATCH, 1)
        results = []
        step = 1. / max(1, int(fps))
        if parse_reset(ident):
            self.__count = 1
            self.__fixed_step = 0
        pbar = ProgressBar(batch)
        for idx in range(batch):
            lin = self.__count
            if not hold:
                if loop > 0:
                    self.__count %= loop
                    lin /= loop
                self.__fixed_step %= fps
            trigger = self.__count % beat == 0
            if passthru is not None:
                trigger = passthru if trigger else None
            results.append([self.__count, lin, self.__fixed_step, trigger])
            if not hold:
                self.__count += 1
                self.__fixed_step += step
            pbar.update_absolute(idx)
        comfy_message(ident, "jovi-tick", {"i": self.__count})
        return list(zip(*results))

class WaveGeneratorNode(JOVBaseNode):
    NAME = "WAVE GENERATOR (JOV) 🌊"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Periodic and Non-Periodic Sinosodials."
    OUTPUT_IS_LIST = (True, True,)
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = (Lexicon.FLOAT, Lexicon.INT, )

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
        return Lexicon._parse(d, JOV_HELP_URL + "/ANIMATE#-wave-generator")

    def run(self, **kw) -> tuple[float, int]:
        op = kw.get(Lexicon.WAVE, [EnumWave.SIN])
        freq = kw.get(Lexicon.FREQ, [1.])
        amp = kw.get(Lexicon.AMP, [1.])
        phase = kw.get(Lexicon.PHASE, [0])
        shift = kw.get(Lexicon.OFFSET, [0])
        delta_time = kw.get(Lexicon.TIME, [0])
        invert = kw.get(Lexicon.INVERT, [False])
        abs = kw.get(Lexicon.ABSOLUTE, [False])
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
        return list(zip(*results))
