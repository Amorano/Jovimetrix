"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

import math
from typing import Any, Tuple

import numpy as np

from comfy.utils import ProgressBar

from Jovimetrix import comfy_message, parse_reset, JOVBaseNode, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.anim import wave_op, EnumWave
from Jovimetrix.sup.util import parse_param, zip_longest_fill, EnumConvertType

# =============================================================================

JOV_CATEGORY = "ANIMATE"

# =============================================================================

class Results(object):
    def __init__(self, *arg, **kw) -> None:
        self.frame = []
        self.lin = []
        self.fixed = []
        self.trigger = []

#

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) ⏱"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (WILDCARD, "FLOAT", "FLOAT", WILDCARD)
    RETURN_NAMES = (Lexicon.VALUE, Lexicon.LINEAR, Lexicon.FPS, Lexicon.TRIGGER)
    OUTPUT_IS_LIST = (True, True, True, True)
    DESCRIPTION = """
The `Tick` node acts as a timer and frame counter, emitting pulses or signals based on time intervals or BPM settings. It allows precise synchronization and control over animation sequences, with options to adjust FPS, BPM, and loop points. This node is useful for generating time-based events or driving animations with rhythmic precision.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            # data to pass on a pulse of the loop
            Lexicon.TRIGGER: (WILDCARD, {"default": None, "tooltip":"Output to send when beat (BPM setting) is hit"}),
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
            Lexicon.STEP: ("INT", {"min": 0, "default": 0, "step": 1, "tooltip": "Steps/Stride between pulses -- useful to do odd or even batches. If set to 0 will stretch from (VAL -> LOOP) / Batch giving a linear range of values."}),
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, cls)

    """
    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")
    """

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        # how many pulses we have done -- total unless reset
        self.__frame = 0

    def run(self, ident, **kw) -> Tuple[int, float, float, Any]:
        passthru = parse_param(kw, Lexicon.TRIGGER, EnumConvertType.ANY, None)[0]
        stride = parse_param(kw, Lexicon.STEP, EnumConvertType.INT, 0, 0)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0)[0]
        self.__frame = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, self.__frame)[0]
        if loop > 0:
            self.__frame %= loop
        self.__frame = max(0, self.__frame)
        hold = parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False)[0]
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1)[0]
        bpm = parse_param(kw, Lexicon.BPM, EnumConvertType.INT, 120, 1)[0]
        divisor = parse_param(kw, Lexicon.NOTE, EnumConvertType.INT, 4, 1)[0]
        beat = 60. / max(1., bpm) / divisor
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 1, 1)[0]
        step_fps = 1. / max(1., float(fps))
        reset = parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)[0]
        if parse_reset(ident) > 0 or reset:
            self.__frame = 0
        trigger = None
        results = Results()
        pbar = ProgressBar(batch)
        step = 1 if stride > 0 else max(1, loop / batch)
        for idx in range(batch):
            trigger = False
            lin = self.__frame if loop == 0 else self.__frame / loop
            fixed_step = math.fmod(self.__frame * step_fps, fps)
            if (math.fmod(fixed_step, beat) == 0):
                trigger = [passthru]
            if loop > 0:
                self.__frame %= loop
            results.frame.append(self.__frame)
            results.lin.append(lin)
            results.fixed.append(float(fixed_step))
            results.trigger.append(trigger)
            if not hold:
                self.__frame += step
            pbar.update_absolute(idx)
        if batch < 2:
            comfy_message(ident, "jovi-tick", {"i": self.__frame})
        return results.frame, results.lin, results.fixed, results.trigger

class WaveGeneratorNode(JOVBaseNode):
    NAME = "WAVE GEN (JOV) 🌊"
    NAME_PRETTY = "WAVE GEN (JOV) 🌊"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = (Lexicon.FLOAT, Lexicon.INT, )
    DESCRIPTION = """
The `Wave Generator` node produces waveforms like sine, square, or sawtooth with adjustable frequency, amplitude, phase, and offset. It's handy for creating oscillating patterns or controlling animation dynamics. This node emits both continuous floating-point values and integer representations of the generated waves.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.WAVE: (EnumWave._member_names_, {"default": EnumWave.SIN.name}),
            Lexicon.FREQ: ("FLOAT", {"default": 1, "min": 0, "step": 0.01, "max": 10000000000000000}),
            Lexicon.AMP: ("FLOAT", {"default": 1, "min": 0, "step": 0.01, "max": 10000000000000000}),
            Lexicon.PHASE: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001, "max": 1.0}),
            Lexicon.OFFSET: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001, "max": 1.0}),
            Lexicon.TIME: ("FLOAT", {"default": 0, "min": 0, "step": 0.000001}),
            Lexicon.INVERT: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[float, int]:
        op = parse_param(kw, Lexicon.WAVE, EnumConvertType.STRING, EnumWave.SIN.name)
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
        return *[[x] for x in zip(*results)],
        # return [list(x) for x in (zip(*results))]
