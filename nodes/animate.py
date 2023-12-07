"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animate
"""

import time
from typing import Any

from Jovimetrix import Logger, JOVBaseNode, WILDCARD, JOV_MAX_DELAY
from Jovimetrix.sup.anim import EnumWaveSimple

# =============================================================================

class TickNode(JOVBaseNode):
    NAME = "🕛 Tick (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"
    DESCRIPTION = "Periodic pulse exporting normalized, delta since last pulse and count."
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", )
    RETURN_NAMES = ("🧮", "🛟", "🕛", "🔺🕛",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {},
            "optional": {
                "total": ("INT", {"min": 0, "default": 0, "step": 1}),
                # forces a MOD on total
                "loop": ("BOOLEAN", {"default": False}),
                # stick the current "count"
                "hold": ("BOOLEAN", {"default": False}),
                # manual total = 0
                "reset": ("BOOLEAN", {"default": False}),
            }}

    @classmethod
    def IS_CHANGED(cls, *arg, **kw) -> Any:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__count = 0
        # previous time, current time
        self.__time = time.perf_counter()
        self.__delta = 0

    def run(self, total: int, loop: bool, hold: bool, reset: bool) -> None:
        if reset:
            self.__count = 0

        # count = self.__count
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

class DelayNode(JOVBaseNode):
    """Delay for some time."""

    NAME = "⏸️ Delay (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"
    DESCRIPTION = "Delay for some time"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🦄",)
    SORT = 70

    @classmethod
    def INPUT_TYPES(cls) -> Any:
        return {"required": {
                    "o": (WILDCARD, {"default": None}),
                    "delay": ("FLOAT", {"step": 0.01, "default" : 0}),
                    "hold": ("BOOLEAN", {"default": False}),
                    "reset": ("BOOLEAN", {"default": False})
                }}

    def __init__(self) -> None:
        self.__delay = 0

    def run(self, o: Any, delay: float, hold: bool, reset: bool) -> dict:
        ''' @TODO
        t = threading.Thread(target=self.__run, daemon=True)
        t.start()
        '''
        if reset:
            self.__delay = 0
            return (self, )

        if hold:
            return(None,)

        if delay != self.__delay:
            self.__delay = delay
            self.__delay = max(0, min(self.__delay, JOV_MAX_DELAY))

        time.sleep(self.__delay)
        return (o,)

    def __run(self) -> None:
        while self.__hold:
            time.sleep(0.1)

class WaveGeneratorNode(JOVBaseNode):
    NAME = "🌊 Wave Generator (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"
    DESCRIPTION = ""
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = ("🛟", "🔟", )
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required":{
                "wave": (EnumWaveSimple._member_names_, {"default": EnumWaveSimple.SIN.name}),
                "phase": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 1.0}),
                "amp": ("FLOAT", {"default": 0.5, "min": 0.0, "step": 0.1}),
                "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 1.0}),
                "max": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 9999.0, "step": 0.05}),
                "frame": ("INT", {"default": 1.0, "min": 0.0, "step": 1.0}),
            }}
        return d

    def run(self, wave: str, phase: float, amp: float, offset: float, max: float, frame: int) -> tuple[float, int]:
        val = 0.
        if (op := WaveGeneratorNode.OP_WAVE.get(wave, None)):
            val = op(phase, amp, offset, max, frame)
        return (val, int(val))

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass