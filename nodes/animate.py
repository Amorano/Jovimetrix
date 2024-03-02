"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

import time

from loguru import logger

import comfy

from Jovimetrix import JOV_HELP_URL, JOVBaseNode
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup import anim
from Jovimetrix.sup.anim import EnumWaveSimple
from Jovimetrix.sup.util import parse_tuple, zip_longest_fill

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"

# =============================================================================

class TickNode(JOVBaseNode):
    NAME = "TICK (JOV) ⏱"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Periodic pulse exporting normalized, delta since last pulse and count."
    INPUT_IS_LIST = False
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT" )
    RETURN_NAMES = (Lexicon.VALUE, Lexicon.LINEAR, Lexicon.TIME, Lexicon.DELTA,
                    f"{Lexicon.NOTE}_128", f"{Lexicon.NOTE}_64",
                    f"{Lexicon.NOTE}_32", f"{Lexicon.NOTE}_16", f"{Lexicon.NOTE}_08",
                    f"{Lexicon.NOTE}_04", f"{Lexicon.NOTE}_02", f"{Lexicon.NOTE}_01" )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.BPM: ("FLOAT", {"min": 1, "max": 60000, "default": 120, "step": 1}),
            # forces a MOD on CYCLE
            Lexicon.LOOP: ("INT", {"min": 0, "default": 0, "step": 1}),
            # stick the current "count"
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            # manual total = 0
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
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
        # logger.debug(float(bpm), beat_04, beat_loop, round(beat_04 * 0.03125))
        beat_128 = beat_loop % round(beat_04 * 0.03125) == 0 and self.__count != 0
        beat_04 = beat_loop % round(beat_04) == 0 and self.__count != 0
        self.__delta = 0
        if not hold:
            # frame up
            self.__count += 1
            self.__delta = (t := time.perf_counter()) - self.__time
            self.__time = t

        return self.__count, lin, self.__time, self.__delta, beat_128, beat_64, beat_32, beat_16, beat_08, beat_04, beat_02, beat_01,

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
            Lexicon.WAVE: (EnumWaveSimple._member_names_, {"default": EnumWaveSimple.SIN.name}),
            Lexicon.FREQ: ("FLOAT", {"default": 1, "min": 0.0, "step": 0.01}),
            Lexicon.AMP: ("FLOAT", {"default": 1, "min": 0.0, "step": 0.01}),
            Lexicon.PHASE: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001}),
            Lexicon.OFFSET: ("FLOAT", {"default": 0, "min": 0.0, "step": 0.001}),
            Lexicon.TIME: ("FLOAT", {"default": 0, "min": 0, "step": 0.000001}),
            Lexicon.BATCH: ("VEC2", {"default": (1, 30), "step": 1, "label": ["COUNT", "FPS"]}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/ANIMATE#-wave-generator")

    def run(self, **kw) -> tuple[float, int]:
        wave = kw.get(Lexicon.WAVE, [EnumWaveSimple.SIN])
        freq = kw.get(Lexicon.FREQ, [1.])
        amp = kw.get(Lexicon.AMP, [1.])
        phase = kw.get(Lexicon.PHASE, [0])
        shift = kw.get(Lexicon.OFFSET, [0])
        delta_time = kw.get(Lexicon.TIME, [0])
        batch = parse_tuple(Lexicon.BATCH, kw, default=(1, 30), clip_min=1)
        results = []
        params = [tuple(x) for x in zip_longest_fill(wave, freq, amp, phase, shift, delta_time, batch)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (wave, freq, amp, phase, shift, delta_time, batch) in enumerate(params):
            val = 0.
            if (op := getattr(anim.Wave, wave.lower(), None)) is not None:
                freq = 1. / freq
                batch_size, batch_fps = batch
                if batch_size == 1:
                    val = op(phase, freq, amp, shift, delta_time)
                    results.append([val, int(val)])
                    continue

                delta = delta_time
                delta_step = 1 / batch_fps
                for _ in range(batch_size):
                    val = op(phase, freq, amp, shift, delta)
                    results.append([val, int(val)])
                    delta += delta_step
            else:
                results.append([val, int(val)])
            pbar.update_absolute(idx)
        return list(zip(*results))
