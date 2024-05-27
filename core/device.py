"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Device -- SYSTEM MAIN -- AUDIO
"""

from typing import Tuple

import torch
from loguru import logger

from Jovimetrix import JOVBaseNode
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.audio import AudioDevice

# =============================================================================

JOV_CATEGORY = "DEVICE"

# =============================================================================

class AudioDeviceNode(JOVBaseNode):
    NAME = "AUDIO DEVICE (JOV) 📺"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ('WAVE',)
    RETURN_NAMES = (Lexicon.WAVE,)
    SORT = 90
    DESCRIPTION = """
The Audio Device node allows you to interact with audio input devices to capture audio data. It provides options to select the audio input device, control automatic recording triggered by the Q system, and manually adjust the recording state. This node enables integration with external audio hardware and facilitates audio data acquisition for processing within the JOVIMETRIX environment.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        dev = AudioDevice()
        dev_list = list(dev.devices.keys())
        d = {
            "required": {},
            "optional": {
                Lexicon.DEVICE: (dev_list, {"default": next(iter(dev_list)), "choice": "system list of audio devices"}),
                Lexicon.TRIGGER: ("BOOLEAN", {"default": True, "tooltip":"Auto-record when executed by the Q"}),
                Lexicon.RECORD: ("BOOLEAN", {"default": True, "tooltip":"Control to manually adjust when the selected device is recording"}),
            }
        }
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        wave = None
        return wave
