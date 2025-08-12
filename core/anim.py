""" Jovimetrix - Animation """

import sys

import numpy as np

from comfy.utils import ProgressBar

from cozy_comfyui import \
    InputType, EnumConvertType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    CozyBaseNode

from cozy_comfyui.maths.ease import \
    EnumEase, \
    ease_op

from cozy_comfyui.maths.norm import \
    EnumNormalize, \
    norm_op

from cozy_comfyui.maths.wave import \
    EnumWave, \
    wave_op

from cozy_comfyui.maths.series import \
    seriesLinear

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

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
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("VALUE", "LINEAR", "EASED", "SCALAR_LIN", "SCALAR_EASE")
    OUTPUT_IS_LIST = (True, True, True, True, True,)
    OUTPUT_TOOLTIPS = (
        "List of values",
        "Normalized values",
        "Eased values",
        "Scalar normalized values",
        "Scalar eased values",
    )
    DESCRIPTION = """
Value generator with normalized values based on based on time interval.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                # forces a MOD on CYCLE
                Lexicon.START: ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize
                }),
                # interval between frames
                Lexicon.STEP: ("FLOAT", {
                    "default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "precision": 3,
                    "tooltip": "Amount to add to each frame per tick"
                }),
                # how many frames to dump....
                Lexicon.COUNT: ("INT", {
                    "default": 1, "min": 1, "max": 1500
                }),
                Lexicon.LOOP: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize,
                    "tooltip": "What value before looping starts. 0 means linear playback (no loop point)"
                }),
                Lexicon.PINGPONG: ("BOOLEAN", {
                    "default": False
                }),
                Lexicon.EASE: (EnumEase._member_names_, {
                    "default": EnumEase.LINEAR.name}),
                Lexicon.NORMALIZE: (EnumNormalize._member_names_, {
                    "default": EnumNormalize.MINMAX2.name}),
                Lexicon.SCALAR: ("FLOAT", {
                    "default": 1, "min": 0, "max": sys.float_info.max
                })

            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[float, ...]:
        """
        Generates a series of numbers with various options including:
        - Custom start value (supporting floating point and negative numbers)
        - Custom step value (supporting floating point and negative numbers)
        - Fixed number of frames
        - Custom loop point (series restarts after reaching this many steps)
        - Ping-pong option (reverses direction at end points)
        - Support for easing functions
        - Normalized output 0..1, -1..1, L2 or ZScore
        """

        start = parse_param(kw, Lexicon.START, EnumConvertType.FLOAT, 0)[0]
        step = parse_param(kw, Lexicon.STEP, EnumConvertType.FLOAT, 0)[0]
        count = parse_param(kw, Lexicon.COUNT, EnumConvertType.INT, 1, 1, 1500)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0, 0)[0]
        pingpong = parse_param(kw, Lexicon.PINGPONG, EnumConvertType.BOOLEAN, False)[0]
        ease = parse_param(kw, Lexicon.EASE, EnumEase, EnumEase.LINEAR.name)[0]
        normalize = parse_param(kw, Lexicon.NORMALIZE, EnumNormalize, EnumNormalize.MINMAX1.name)[0]
        scalar = parse_param(kw, Lexicon.SCALAR, EnumConvertType.FLOAT, 1, 0)[0]

        if step == 0:
            step = 1

        cycle = seriesLinear(start, step, count, loop, pingpong)
        linear = norm_op(normalize, np.array(cycle))
        eased = ease_op(ease, linear, len(linear))
        scalar_linear = linear * scalar
        scalar_eased = eased * scalar

        return (
            cycle,
            linear.tolist(),
            eased.tolist(),
            scalar_linear.tolist(),
            scalar_eased.tolist(),
        )

class WaveGeneratorNode(CozyBaseNode):
    NAME = "WAVE GEN (JOV) 🌊"
    NAME_PRETTY = "WAVE GEN (JOV) 🌊"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = ("FLOAT", "INT", )
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
                Lexicon.FREQ: ("FLOAT", {
                    "default": 1, "min": 0, "max": sys.float_info.max, "step": 0.01,}),
                Lexicon.AMP: ("FLOAT", {
                    "default": 1, "min": 0, "max": sys.float_info.max, "step": 0.01,}),
                Lexicon.PHASE: ("FLOAT", {
                    "default": 0, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.OFFSET: ("FLOAT", {
                    "default": 0, "min": 0, "max": 1, "step": 0.001}),
                Lexicon.TIME: ("FLOAT", {
                    "default": 0, "min": 0, "max": sys.float_info.max, "step": 0.0001}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False}),
                Lexicon.ABSOLUTE: ("BOOLEAN", {
                    "default": False,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[float, int]:
        op = parse_param(kw, Lexicon.WAVE, EnumWave, EnumWave.SIN.name)
        freq = parse_param(kw, Lexicon.FREQ, EnumConvertType.FLOAT, 1, 0)
        amp = parse_param(kw, Lexicon.AMP, EnumConvertType.FLOAT, 1, 0)
        phase = parse_param(kw, Lexicon.PHASE, EnumConvertType.FLOAT, 0, 0)
        shift = parse_param(kw, Lexicon.OFFSET, EnumConvertType.FLOAT, 0, 0)
        delta_time = parse_param(kw, Lexicon.TIME, EnumConvertType.FLOAT, 0, 0)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        absolute = parse_param(kw, Lexicon.ABSOLUTE, EnumConvertType.BOOLEAN, False)
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
            val = max(-sys.float_info.max, min(val, sys.float_info.max))
            results.append([val, int(val)])
            pbar.update_absolute(idx)
        return *list(zip(*results)),

'''
class TickOldNode(CozyBaseNode):
    NAME = "TICK OLD (JOV) ⏱"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", COZY_TYPE_ANY, COZY_TYPE_ANY,)
    RETURN_NAMES = ("VAL", "LINEAR", "FPS", "TRIGGER", "BATCH",)
    OUTPUT_IS_LIST = (True, False, False, False, False,)
    OUTPUT_TOOLTIPS = (
        "Current value for the configured tick as ComfyUI List",
        "Normalized tick value (0..1) based on BPM and Loop",
        "Current 'frame' in the tick based on FPS setting",
        "Based on the BPM settings, on beat hit, output the input at '⚡'",
        "Current batch of values for the configured tick as standard list which works in other Jovimetrix nodes",
    )
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
                    "tooltip": "Output to send when beat (BPM setting) is hit"
                }),
                # forces a MOD on CYCLE
                Lexicon.START: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize,
                }),
                Lexicon.LOOP: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize,
                    "tooltip": "Number of frames before looping starts. 0 means continuous playback (no loop point)"
                }),
                Lexicon.FPS: ("INT", {
                    "default": 24, "min": 1
                }),
                Lexicon.BPM: ("INT", {
                    "default": 120, "min": 1, "max": 60000,
                    "tooltip": "BPM trigger rate to send the input. If input is empty, TRUE is sent on trigger"
                }),
                Lexicon.NOTE: ("INT", {
                    "default": 4, "min": 1, "max": 256,
                    "tooltip": "Number of beats per measure. Quarter note is 4, Eighth is 8, 16 is 16, etc."}),
                # how many frames to dump....
                Lexicon.BATCH: ("INT", {
                    "default": 1, "min": 1, "max": 32767,
                    "tooltip": "Number of frames wanted"
                }),
                Lexicon.STEP: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize
                }),
            }
        })
        return Lexicon._parse(d)

    def run(self, ident, **kw) -> tuple[int, float, float, Any]:
        passthru = parse_param(kw, Lexicon.TRIGGER, EnumConvertType.ANY, None)[0]
        stride = parse_param(kw, Lexicon.STEP, EnumConvertType.INT, 0)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0)[0]
        start = parse_param(kw, Lexicon.START, EnumConvertType.INT, self.__frame)[0]
        if loop != 0:
            self.__frame %= loop
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1)[0]
        bpm = parse_param(kw, Lexicon.BPM, EnumConvertType.INT, 120, 1)[0]
        divisor = parse_param(kw, Lexicon.NOTE, EnumConvertType.INT, 4, 1)[0]
        beat = 60. / max(1., bpm) / divisor
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 1, 1)[0]
        step_fps = 1. / max(1., float(fps))

        trigger = None
        results = ResultObject()
        pbar = ProgressBar(batch)
        step = stride if stride != 0 else max(1, loop / batch)
        for idx in range(batch):
            trigger = False
            lin = start if loop == 0 else start / loop
            fixed_step = math.fmod(start * step_fps, fps)
            if (math.fmod(fixed_step, beat) == 0):
                trigger = [passthru]
            if loop != 0:
                start %= loop
            results.frame.append(start)
            results.lin.append(float(lin))
            results.fixed.append(float(fixed_step))
            results.trigger.append(trigger)
            results.batch.append(start)
            start += step
            pbar.update_absolute(idx)

        return (results.frame, results.lin, results.fixed, results.trigger, results.batch,)

'''