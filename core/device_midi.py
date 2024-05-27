"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Device -- MIDI

    type 0 (single track): all messages are saved in one track
    type 1 (synchronous): all tracks start at the same time
    type 2 (asynchronous): each track is independent of the others
"""

from typing import Tuple
from math import isclose
from queue import Queue

from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOVBaseNode
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import EnumConvertType, parse_param
from Jovimetrix.sup.midi import midi_device_names, \
    MIDIMessage, MIDINoteOnFilter, MIDIServerThread

# =============================================================================

JOV_CATEGORY = "DEVICE"

# =============================================================================

class MIDIMessageNode(JOVBaseNode):
    NAME = "MIDI MESSAGE (JOV) 🎛️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.VALUE, Lexicon.NORMALIZE, )
    SORT = 10
    DESCRIPTION = """
The MIDI Message node processes MIDI messages received from an external MIDI controller or device. It accepts MIDI messages as input and returns various attributes of the MIDI message, including whether the message is valid, the MIDI channel, control number, note number, value, and normalized value. This node is useful for integrating MIDI control into creative projects, allowing users to respond to MIDI input in real-time.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
                "required": {} ,
                "optional": {
                Lexicon.MIDI: ('JMIDIMSG', {"default": None})
            }
        }
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[object, bool, int, int, int, float, float]:
        message = parse_param(kw, Lexicon.MIDI, EnumConvertType.ANY, None)
        results = []
        pbar = ProgressBar(len(message))
        for idx, (message,) in enumerate(message):
            data = [message]
            if message is None:
                data.extend([False, -1, -1, -1, -1, -1])
            else:
                data.extend(*message.flat)
            results.append(data)
            pbar.update_absolute(idx)
        return (results,)

class MIDIReaderNode(JOVBaseNode):
    NAME = "MIDI READER (JOV) 🎹"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT',)
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.VALUE, Lexicon.NORMALIZE,)
    SORT = 5
    DEVICES = midi_device_names()
    DESCRIPTION = """
The MIDI Reader node captures MIDI messages from an external MIDI device or controller. It monitors MIDI input and provides information about the received MIDI messages, including whether a note is being played, the MIDI channel, control number, note number, value, and a normalized value. This node is essential for integrating MIDI control into various applications, such as music production, live performances, and interactive installations.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
                "required": {} ,
                "optional": {
                Lexicon.DEVICE : (cls.DEVICES, {"default": cls.DEVICES[0] if len(cls.DEVICES) > 0 else None})
            }
        }
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__q_in = Queue()
        self.__device = None
        self.__note = 0
        self.__note_on = False
        self.__channel = 0
        self.__control = 0
        self.__value = 0
        self.__SERVER = MIDIServerThread(self.__q_in, None, self.__process, daemon=True)
        self.__SERVER.start()

    def __process(self, data) -> None:
        self.__channel = data.channel
        self.__note = 0
        self.__control = 0
        self.__note_on = False
        match data.type:
            case "control_change":
                # control=8 value=14 time=0
                self.__control = data.control
                self.__value = data.value
            case "note_on":
                self.__note = data.note
                self.__note_on = True
                self.__value = data.velocity
            case "note_off":
                self.__note = data.note
                self.__value = data.velocity

    def run(self, **kw) -> Tuple[MIDIMessage, bool, int, int, int, int, float]:
        device = parse_param(kw, Lexicon.DEVICE, EnumConvertType.STRING, None)[0]
        if device != self.__device:
            self.__q_in.put(device)
            self.__device = device
        normalize = self.__value / 127.
        msg = MIDIMessage(self.__note_on, self.__channel, self.__control, self.__note, self.__value)
        return (msg, self.__note_on, self.__channel, self.__control, self.__note, self.__value, normalize, )

class MIDIFilterEZNode(JOVBaseNode):
    NAME = "MIDI FILTER EZ (JOV) ❇️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN',)
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.TRIGGER,)
    SORT = 25
    DESCRIPTION = """
The MIDI Filter EZ node allows you to filter MIDI messages based on various criteria, including MIDI mode (such as note on or note off), MIDI channel, control number, note number, value, and normalized value. This node is useful for processing MIDI input and selectively passing through only the desired messages. It helps simplify MIDI data handling by allowing you to focus on specific types of MIDI events.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {} ,
            "optional": {
                Lexicon.MIDI: ('JMIDIMSG', {"default": None}),
                Lexicon.MODE: (MIDINoteOnFilter._member_names_, {"default": MIDINoteOnFilter.IGNORE.name}),
                Lexicon.CHANNEL: ("INT", {"default": -1, "min": -1, "max": 127, "step": 1}),
                Lexicon.CONTROL: ("INT", {"default": -1, "min": -1, "max": 127, "step": 1}),
                Lexicon.NOTE: ("INT", {"default": -1, "min": -1, "max": 127, "step": 1}),
                Lexicon.VALUE: ("INT", {"default": -1, "min": -1, "max": 127, "step": 1}),
                Lexicon.NORMALIZE: ("FLOAT", {"default": -1, "min": -1, "max": 1, "step": 0.01})
            }
        }
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[MIDIMessage, bool]:
        message = parse_param(kw, Lexicon.MIDI, EnumConvertType.ANY, None)[0]
        if message is None:
            logger.warning('no midi message. connected?')
            return (None, False, )

        # empty values mean pass-thru (no filter)
        val = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, MIDINoteOnFilter.IGNORE.name)[0]
        val = MIDINoteOnFilter[val]
        if val != MIDINoteOnFilter.IGNORE:
            if val == MIDINoteOnFilter.NOTE_ON and message.note_on != True:
                return (message, False, )
            if val == MIDINoteOnFilter.NOTE_OFF and message.note_on != False:
                return (message, False, )

        if (val := parse_param(kw, Lexicon.CHANNEL, EnumConvertType.INT, -1)[0]) != -1 and val != message.channel:
            return (message, False, )
        if (val := parse_param(kw, Lexicon.CONTROL, EnumConvertType.INT, -1)[0]) != -1 and val != message.control:
            return (message, False, )
        if (val := parse_param(kw, Lexicon.NOTE, EnumConvertType.INT, -1)[0]) != -1 and val != message.note:
            return (message, False, )
        if (val := parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, -1)[0]) != -1 and val != message.value:
            return (message, False, )
        if (val := parse_param(kw, Lexicon.NORMALIZE, EnumConvertType.INT, -1)[0]) != -1 and isclose(message.normal):
            return (message, False, )
        return (message, True, )

class MIDIFilterNode(JOVBaseNode):
    NAME = "MIDI FILTER (JOV) ✳️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.TRIGGER,)
    SORT = 20
    EPSILON = 1e-6
    DESCRIPTION = """
The MIDI Filter node provides advanced filtering capabilities for MIDI messages based on various criteria, including MIDI mode (such as note on or note off), MIDI channel, control number, note number, value, and normalized value. It allows you to filter out unwanted MIDI events and selectively process only the desired ones. This node offers flexibility in MIDI data processing, enabling precise control over which MIDI messages are passed through for further processing.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {} ,
            "optional": {
                Lexicon.MIDI: ('JMIDIMSG', {"default": None}),
                Lexicon.ON: (MIDINoteOnFilter._member_names_, {"default": MIDINoteOnFilter.IGNORE.name}),
                Lexicon.CHANNEL: ("STRING", {"default": ""}),
                Lexicon.CONTROL: ("STRING", {"default": ""}),
                Lexicon.NOTE: ("STRING", {"default": ""}),
                Lexicon.VALUE: ("STRING", {"default": ""}),
                Lexicon.NORMALIZE: ("STRING", {"default": ""})
            }
        }
        return Lexicon._parse(d, cls)

    def __filter(self, data: str, value: float) -> bool:
        if not data:
            return True
        """
        parse string blocks of "numbers" into range(s) to compare. e.g.:

        1
        5-10
        2

        Would check == 1, == 2 and 5 <= x <= 10
        """
        # can you use float for everything to compare?

        try:
            value = float(value)
        except Exception as e:
            value = float("nan")
            logger.error(str(e))

        for line in data.split(','):
            if len(a_range := line.split('-')) > 1:
                try:
                    a, b = a_range[:2]
                    if float(a) <= value <= float(b):
                        return True
                except Exception as e:
                    logger.error(str(e))

            try:
                if isclose(value, float(line)):
                    return True
            except Exception as e:
                logger.error(str(e))
        return False

    def run(self, **kw) -> Tuple[bool]:
        message = parse_param(kw, Lexicon.MIDI, EnumConvertType.ANY, None)[0]
        if message is None:
            logger.warning('no midi message. connected?')
            return (message, False, )

        # empty values mean pass-thru (no filter)
        val = parse_param(kw, Lexicon.ON, EnumConvertType.STRING, MIDINoteOnFilter.IGNORE.name)[0]
        val = MIDINoteOnFilter[val]
        if val != MIDINoteOnFilter.IGNORE:
            if val == "TRUE" and message.note_on != True:
                return (message, False, )
            if val == "FALSE" and message.note_on != False:
                return (message, False, )

        if self.__filter(message.channel, parse_param(kw, Lexicon.CHANNEL, EnumConvertType.BOOLEAN, False)[0]) == False:
            return (message, False, )
        if self.__filter(message.control, parse_param(kw, Lexicon.CONTROL, EnumConvertType.BOOLEAN, False)[0]) == False:
            return (message, False, )
        if self.__filter(message.note, parse_param(kw, Lexicon.NOTE, EnumConvertType.BOOLEAN, False)[0]) == False:
            return (message, False, )
        if self.__filter(message.value, parse_param(kw, Lexicon.VALUE, EnumConvertType.BOOLEAN, False)[0]) == False:
            return (message, False, )
        if self.__filter(message.normal, parse_param(kw, Lexicon.NORMALIZE, EnumConvertType.BOOLEAN, False)[0]) == False:
            return (message, False, )
        return (message, True, )
