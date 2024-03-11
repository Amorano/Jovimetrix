"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

import time

from loguru import logger

import comfy

from Jovimetrix import JOV_HELP_URL, WILDCARD, JOVBaseNode, parse_reset
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.anim import EnumWave, wave_op
from Jovimetrix.sup.util import parse_tuple, zip_longest_fill

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"

# =============================================================================

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) ⏱"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Periodic pulse exporting normalized, delta since last pulse and count."
    INPUT_IS_LIST = False
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", )
    RETURN_NAMES = (Lexicon.VALUE, Lexicon.LINEAR, Lexicon.TIME, Lexicon.DELTA,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            # forces a MOD on CYCLE
            Lexicon.LOOP: ("INT", {"min": 0, "default": 0, "step": 1}),
            # stick the current "count"
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            # manual total = 0
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
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
        self.__count = 0
        # previous time, current time
        self.__time = time.perf_counter()
        self.__delta = 0

    def run(self, ident, **kw) -> tuple[int, float, float, float]:
        loop = kw.get(Lexicon.LOOP, 0)
        hold = kw.get(Lexicon.WAIT, False)
        if parse_reset(ident):
            self.__time = time.perf_counter()
            self.__count = 0
        if loop > 0:
            self.__count %= loop
        lin = self.__count / (loop if loop != 0 else 1)
        self.__delta = 0
        if not hold:
            self.__count += 1
            self.__delta = (t := time.perf_counter()) - self.__time
            self.__time = t
        return self.__count, lin, self.__time, self.__delta,

class Pulsetronome(JOVBaseNode):
    NAME = "PULSETRONOME (JOV) 🥁"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Sends trigger pulse when the specific beat filter is matched"
    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = (Lexicon.NOTE,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.FLOAT: ("FLOAT", {"default": 0, "min": 0, "step": 0.1, "forceInput": True, "tooltip": "Current time index for calculating the beat modulation"}),
            Lexicon.BPM: ("FLOAT", {"min": 1, "max": 60000, "default": 120, "step": 1}),
            Lexicon.NOTE: ("INT", {"default": 4, "min": 1, "max": 256, "step": 1,
                                   "tooltip":"Number of beats per measure. Quarter note is 4, Eighth is 8, 16 is 16, etc..."}),
            Lexicon.ANY: (WILDCARD, {"default": None, "tooltip":"Output to send on trigger. Can be anything."}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/ANIMATE#-pulsetronome")

    def run(self, **kw) -> tuple[float]:
        beats = []
        index = kw.get(Lexicon.VALUE, [0])
        bpm = kw.get(Lexicon.BPM, [120])
        divisor = kw.get(Lexicon.NOTE, [4])
        params = [tuple(x) for x in zip_longest_fill(index, bpm, divisor)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (index, bpm, divisor) in enumerate(params):
            beat = 240000. / bpm
            val = int(index) % round(beat / divisor) == 0
            beats.append([val])
            pbar.update_absolute(idx)
        return list(zip(*beats))

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
            Lexicon.BATCH: ("VEC2", {"default": (1, 30), "step": 1, "label": ["COUNT", "FPS"],
                                     "tooltip": "Number of frames wanted; Playback rate (FPS)"}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/ANIMATE#-wave-generator")

    def run(self, **kw) -> tuple[float, int]:
        op = kw.get(Lexicon.WAVE, [EnumWave.SIN])
        freq = kw.get(Lexicon.FREQ, [1.])
        amp = kw.get(Lexicon.AMP, [1.])
        phase = kw.get(Lexicon.PHASE, [0])
        shift = kw.get(Lexicon.OFFSET, [0])
        delta_time = kw.get(Lexicon.TIME, [0])
        batch = parse_tuple(Lexicon.BATCH, kw, default=(1, 30), clip_min=1)
        results = []
        params = [tuple(x) for x in zip_longest_fill(op, freq, amp, phase, shift, delta_time, batch)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (op, freq, amp, phase, shift, delta_time, batch) in enumerate(params):
            val = 0.
            freq = 1. / freq
            batch_size, batch_fps = batch
            if batch_size == 1:
                val = wave_op(op, phase, freq, amp, shift, delta_time)
                results.append([val, int(val)])
                continue

            delta = delta_time
            delta_step = 1 / batch_fps
            for _ in range(batch_size):
                val = wave_op(op, phase, freq, amp, shift, delta)
                results.append([val, int(val)])
                delta += delta_step
            pbar.update_absolute(idx)
        return list(zip(*results))
