"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

import time

from loguru import logger

import comfy

from Jovimetrix import JOVBaseNode, IT_REQUIRED
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import deep_merge_dict
from Jovimetrix.sup import anim
from Jovimetrix.sup.anim import EnumWaveSimple

# =============================================================================

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) ⏱"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"
    DESCRIPTION = "Periodic pulse exporting normalized, delta since last pulse and count."
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT" )
    RETURN_NAMES = (Lexicon.COUNT, Lexicon.LINEAR, Lexicon.TIME, Lexicon.DELTA,
                    f"{Lexicon.NOTE}_128", f"{Lexicon.NOTE}_64",
                    f"{Lexicon.NOTE}_32", f"{Lexicon.NOTE}_16", f"{Lexicon.NOTE}_08",
                    f"{Lexicon.NOTE}_04", f"{Lexicon.NOTE}_02", f"{Lexicon.NOTE}_01" )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.BPM: ("FLOAT", {"min": 1, "max": 60000, "default": 120, "step": 1}),
                # forces a MOD on CYCLE
                Lexicon.LOOP: ("INT", {"min": 0, "default": 0, "step": 1}),
                # stick the current "count"
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
                # manual total = 0
                Lexicon.RESET: ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_REQUIRED, d)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__count = 0
        # previous time, current time
        self.__time = time.perf_counter()
        self.__delta = 0

    def run(self, **kw) -> tuple[int, float, float, float, float, float, float, float, float, float, float, float, float]:
        bpm = kw.get(Lexicon.BPM, 1.)
        loop = kw.get(Lexicon.LOOP, 0)
        hold = kw.get(Lexicon.WAIT, False)
        reset = kw.get(Lexicon.RESET, False)

        if reset:
            self.__count = 0

        if loop > 0:
            self.__count %= loop
        lin = self.__count / (loop if loop != 0 else 1)

        beat_04 = 60000. / float(bpm)
        beat_loop = float(self.__count)
        beat_01 = beat_loop % round(beat_04 * 4) == 0 and self.__count != 0
        beat_02 = beat_loop % round(beat_04 * 2) == 0 and self.__count != 0
        beat_08 = beat_loop % round(beat_04 * 0.5) == 0 and self.__count != 0
        beat_16 = beat_loop % round(beat_04 * 0.25) == 0 and self.__count != 0
        beat_32 = beat_loop % round(beat_04 * 0.125) == 0 and self.__count != 0
        beat_64 = beat_loop % round(beat_04 * 0.0625) == 0 and self.__count != 0
        # print(float(bpm), beat_04, beat_loop, round(beat_04 * 0.03125))
        beat_128 = beat_loop % round(beat_04 * 0.03125) == 0 and self.__count != 0
        beat_04 = beat_loop % round(beat_04) == 0 and self.__count != 0
        self.__delta = 0
        if not hold:
            # frame up
            self.__count += 1
            self.__delta = (t := time.perf_counter()) - self.__time
            self.__time = t

        return (self.__count, lin, self.__time, self.__delta, beat_128, beat_64, beat_32, beat_16, beat_08, beat_04, beat_02, beat_01, )

class WaveGeneratorNode(JOVBaseNode):
    NAME = "WAVE GENERATOR (JOV) 🌊"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"
    DESCRIPTION = ""
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = (Lexicon.FLOAT, Lexicon.INT, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional":{
                Lexicon.WAVE: (EnumWaveSimple._member_names_, {"default": EnumWaveSimple.SIN.name}),
                Lexicon.FREQ: ("FLOAT", {"default": 1, "min": 0.0, "step": 0.01}),
                Lexicon.AMP: ("FLOAT", {"default": 1, "min": 0.0, "step": 0.01}),
                Lexicon.PHASE: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001}),
                Lexicon.SHIFT: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001}),
                Lexicon.TIME: ("FLOAT", {"default": 0, "min": 0, "step": 0.000001}),
            }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[float, int]:
        val = 0.
        wave = kw.get(Lexicon.WAVE, EnumWaveSimple.SIN)
        freq = kw.get(Lexicon.FREQ, 1.)
        amp = kw.get(Lexicon.AMP, 1.)
        phase = kw.get(Lexicon.PHASE, 0)
        shift = kw.get(Lexicon.SHIFT, 0)
        delta_time = kw.get(Lexicon.TIME, 0)
        if (op := getattr(anim.Wave, wave.lower(), None)) is not None:
            val = op(phase, freq, amp, shift, delta_time)
        return (val, int(val))
