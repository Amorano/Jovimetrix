"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

import time
from typing import Any

from Jovimetrix import deep_merge_dict, \
    Logger, JOVBaseNode, Lexicon, \
    IT_REQUIRED
from Jovimetrix.sup.anim import EnumWaveSimple

# =============================================================================

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) 🕛"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"
    DESCRIPTION = "Periodic pulse exporting normalized, delta since last pulse and count."
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", )
    RETURN_NAMES = (Lexicon.COUNT, Lexicon.LINEAR, Lexicon.TIME, Lexicon.DELTA_TIME,)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.AMT: ("INT", {"min": 0, "default": 0, "step": 1}),
                # forces a MOD on total
                Lexicon.LOOP: ("BOOLEAN", {"default": False}),
                # stick the current "count"
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
                # manual total = 0
                Lexicon.RESET: ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_REQUIRED, d)

    @classmethod
    def IS_CHANGED(cls, *arg, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__count = 0
        # previous time, current time
        self.__time = time.perf_counter()
        self.__delta = 0

    def run(self, **kw) -> tuple[int, float, float, float]:
        total = kw.get(Lexicon.AMT, 0)
        loop = kw.get(Lexicon.LOOP, 0)
        hold = kw.get(Lexicon.WAIT, 0)
        reset = kw.get(Lexicon.RESET, 0)

        if reset:
            self.__count = 0

        if loop and total > 0:
            self.__count %= total
        lin = (self.__count / total) if total != 0 else 1

        t = self.__time
        if not hold:
            self.__count += 1
            t = time.perf_counter()

        self.__delta = t - self.__time
        self.__time = t

        return (self.__count, lin, t, self.__delta,)

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
                Lexicon.PHASE: ("FLOAT", {"default": 1, "min": 0.0, "step": 0.01}),
                Lexicon.AMP: ("FLOAT", {"default": 1, "min": 0.0, "step": 0.1}),
                Lexicon.OFFSET: ("FLOAT2", {"default": (0, 0), "min": 0.0, "step": 0.1}),
                Lexicon.MAX: ("FLOAT", {"default": 0.5, "min": 0.0, "max": 9999.0, "step": 0.01}),
                Lexicon.FRAME: ("INT", {"default": 1, "min": 0, "step": 1}),
            }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[float, int]:
        val = 0.
        wave = kw.get(Lexicon.WAVE, EnumWaveSimple.SIN)
        phase = kw.get(Lexicon.PHASE, 1)
        amp = kw.get(Lexicon.AMP, 1)
        offset = kw.get(Lexicon.OFFSET, 0)
        maximum = kw.get(Lexicon.MAX, 0.5)
        frame = kw.get(Lexicon.FRAME, 1)

        if (op := WaveGeneratorNode.OP_WAVE.get(wave, None)):
            val = op(phase, amp, offset, maximum, frame)
        return (val, int(val))

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass