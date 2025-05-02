""" Jovimetrix - Animation """

import sys
import math
from typing import Any

import numpy as np

from comfy.utils import ProgressBar

from cozy_comfyui import \
    InputType, EnumConvertType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.node import \
    COZY_TYPE_ANY, \
    CozyBaseNode

from cozy_comfyui.api import \
    comfy_api_post, parse_reset

from .. import \
    Lexicon

from ..sup.anim import \
    EnumWave, \
    wave_op

JOV_CATEGORY = "ANIMATION"

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ResultObject(object):
    def __init__(self, *arg, **kw) -> None:
        self.frame = []
        self.lin = []
        self.fixed = []
        self.trigger = []
        self.batch = []

class TickNode(CozyBaseNode):
    NAME = "TICK (JOV) ⏱"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", COZY_TYPE_ANY, COZY_TYPE_ANY,)
    RETURN_NAMES = ("VAL", Lexicon.LINEAR, Lexicon.FPS, Lexicon.TRIGGER, "BATCH",)
    OUTPUT_IS_LIST = (True, False, False, False, False,)
    OUTPUT_TOOLTIPS = (
        "Current value for the configured tick as ComfyUI List",
        "Normalized tick value (0..1) based on BPM and Loop",
        "Current 'frame' in the tick based on FPS setting",
        "Based on the BPM settings, on beat hit, output the input at '⚡'",
        "Current batch of values for the configured tick as standard list which works in other Jovimetrix nodes",
    )
    SORT = 50
    DESCRIPTION = """
A timer and frame counter, emitting pulses or signals based on time intervals. It allows precise synchronization and control over animation sequences, with options to adjust FPS, BPM, and loop points. This node is useful for generating time-based events or driving animations with rhythmic precision.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                # data to pass on a pulse of the loop
                Lexicon.TRIGGER: (COZY_TYPE_ANY, {
                    "default": None,
                    "tooltip":"Output to send when beat (BPM setting) is hit"
                }),
                # forces a MOD on CYCLE
                "VALUE": ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize,
                    "tooltip": "the current frame number of the tick"
                }),
                Lexicon.LOOP: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize,
                    "tooltip": "number of frames before looping starts. 0 means continuous playback (no loop point)"
                }),
                Lexicon.FPS: ("INT", {
                    "default": 24, "min": 1,
                    "tooltip": "Fixed frame step rate based on FPS (1/FPS)"
                }),
                "BPM": ("INT", {
                    "default": 120, "min": 1, "max": 60000,
                    "tooltip": "BPM trigger rate to send the input. If input is empty, TRUE is sent on trigger"
                }),
                Lexicon.NOTE: ("INT", {
                    "default": 4, "min": 1, "max": 256,
                                    "tooltip":"Number of beats per measure. Quarter note is 4, Eighth is 8, 16 is 16, etc."}),
                # stick the current "count"
                Lexicon.WAIT: ("BOOLEAN", {
                    "default": False}),
                # manual total = 0
                "RESET": ("BOOLEAN", {"default": False}),
                # how many frames to dump....
                "BATCH": ("INT", {
                    "default": 1, "min": 1, "max": 32767,
                    "tooltip": "Number of frames wanted"
                }),
                Lexicon.STEP: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize
                }),
            }
        })
        return Lexicon._parse(d)

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        # how many pulses we have done -- total unless reset
        self.__frame = 0

    def run(self, ident, **kw) -> tuple[int, float, float, Any]:
        passthru = parse_param(kw, Lexicon.TRIGGER, EnumConvertType.ANY, None)[0]
        stride = parse_param(kw, Lexicon.STEP, EnumConvertType.INT, 0, 0, sys.maxsize)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0, 0, sys.maxsize)[0]
        self.__frame = parse_param(kw, "VALUE", EnumConvertType.INT, self.__frame, 0, sys.maxsize)[0]
        if loop != 0:
            self.__frame %= loop
        # start_frame = max(0, start_frame)
        hold = parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False)[0]
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1)[0]
        bpm = parse_param(kw, "BPM", EnumConvertType.INT, 120, 1)[0]
        divisor = parse_param(kw, Lexicon.NOTE, EnumConvertType.INT, 4, 1)[0]
        beat = 60. / max(1., bpm) / divisor
        batch = parse_param(kw, "BATCH", EnumConvertType.INT, 1, 1)[0]
        step_fps = 1. / max(1., float(fps))
        reset = parse_param(kw, "RESET", EnumConvertType.BOOLEAN, False)[0]
        if loop == 0 and (parse_reset(ident) > 0 or reset):
            self.__frame = 0
        trigger = None
        results = ResultObject()
        pbar = ProgressBar(batch)
        step = stride if stride != 0 else max(1, loop / batch)
        for idx in range(batch):
            trigger = False
            lin = self.__frame if loop == 0 else self.__frame / loop
            fixed_step = math.fmod(self.__frame * step_fps, fps)
            if (math.fmod(fixed_step, beat) == 0):
                trigger = [passthru]
            if loop != 0:
                self.__frame %= loop
            results.frame.append(self.__frame)
            results.lin.append(float(lin))
            results.fixed.append(float(fixed_step))
            results.trigger.append(trigger)
            results.batch.append(self.__frame)
            if not hold:
                self.__frame += step
            pbar.update_absolute(idx)

        if batch < 2:
            comfy_api_post("jovi-tick", ident, {"i": self.__frame})
        return (results.frame, results.lin, results.fixed, results.trigger, results.batch,)

class TickSimpleNode(CozyBaseNode):
    NAME = "TICK SIMPLE (JOV) ⏱"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("VALUE", "LINEAR")
    OUTPUT_IS_LIST = (True, True,)
    OUTPUT_TOOLTIPS = (
        "List of values",
        "Normalized values (0..1)",
    )
    SORT = 55
    DESCRIPTION = """
Value generator with normalized values based on based on time interval.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                # forces a MOD on CYCLE
                "VALUE": ("INT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Starting value of the tick"
                }),
                # interval between frames
                "STEP": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize, "precision": 3,
                    "tooltip": "Amount to add to each frame per tick"
                }),
                "LOOP": ("INT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "What value before looping starts. 0 means linear playback (no loop point)"
                }),
                # how many frames to dump....
                "BATCH": ("INT", {
                    "default": 1, "min": 1, "max": 1500,
                    "tooltip": "Total frames wanted"
                }),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[int, float|int]:
        value = parse_param(kw, "VALUE", EnumConvertType.INT, 0, -sys.maxsize, sys.maxsize)[0]
        step = parse_param(kw, "STEP", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)[0]
        loop = parse_param(kw, "LOOP", EnumConvertType.INT, 0, -sys.maxsize, sys.maxsize)[0]
        batch = parse_param(kw, "BATCH", EnumConvertType.INT, 1, 1, 1500)[0]
        if loop == 0:
            loop = batch

        results = []
        current = float(value)
        step = step or 1.0
        pbar = ProgressBar(batch)
        for idx in range(0, batch):
            wrapped = (current - value) % loop + value if loop else current
            lin = (wrapped - value) / loop if loop else 0
            results.append([round(wrapped, 6), round(lin, 6)])
            current += step
            pbar.update_absolute(idx)
        return *list(zip(*results)),

class WaveGeneratorNode(CozyBaseNode):
    NAME = "WAVE GEN (JOV) 🌊"
    NAME_PRETTY = "WAVE GEN (JOV) 🌊"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = (Lexicon.FLOAT, Lexicon.INT, )
    SORT = 90
    DESCRIPTION = """
Produce waveforms like sine, square, or sawtooth with adjustable frequency, amplitude, phase, and offset. It's handy for creating oscillating patterns or controlling animation dynamics. This node emits both continuous floating-point values and integer representations of the generated waves.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.WAVE: (EnumWave._member_names_, {
                    "default": EnumWave.SIN.name}),
                "FREQ": ("FLOAT", {
                    "default": 1, "min": 0, "max": sys.maxsize, "step": 0.01,
                    "tooltip": "Frequency"}),
                Lexicon.AMP: ("FLOAT", {
                    "default": 1, "min": 0, "max": sys.maxsize, "step": 0.01}),
                "PHASE": ("FLOAT", {
                    "default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "OFFSET": ("FLOAT", {
                    "default": 0, "min": 0.0, "max": 1.0, "step": 0.001}),
                Lexicon.TIME: ("FLOAT", {
                    "default": 0, "min": 0, "max": sys.maxsize, "step": 0.0001}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False}),
                "ABSOLUTE": ("BOOLEAN", {
                    "default": False,
                    "tooltips": "Return the absolute value of the input"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[float, int]:
        op = parse_param(kw, Lexicon.WAVE, EnumWave, EnumWave.SIN.name)
        freq = parse_param(kw, "FREQ", EnumConvertType.FLOAT, 1., 0.000001, sys.maxsize)
        amp = parse_param(kw, Lexicon.AMP, EnumConvertType.FLOAT, 1., 0., sys.maxsize)
        phase = parse_param(kw, "PHASE", EnumConvertType.FLOAT, 0.)
        shift = parse_param(kw, "OFFSET", EnumConvertType.FLOAT, 0.)
        delta_time = parse_param(kw, Lexicon.TIME, EnumConvertType.FLOAT, 0., 0., sys.maxsize)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        absolute = parse_param(kw, "ABSOLUTE", EnumConvertType.BOOLEAN, False)
        results = []
        params = list(zip_longest_fill(op, freq, amp, phase, shift, delta_time, invert, absolute))
        pbar = ProgressBar(len(params))
        for idx, (op, freq, amp, phase, shift, delta_time, invert, absolute) in enumerate(params):
            # freq = 1. / freq
            if invert:
                amp = 1. / val
            val = wave_op(op, phase, freq, amp, shift, delta_time)
            if absolute:
                val = np.abs(val)
            val = max(-sys.maxsize, min(val, sys.maxsize))
            results.append([val, int(val)])
            pbar.update_absolute(idx)
        return *list(zip(*results)),
