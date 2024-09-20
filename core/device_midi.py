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

from Jovimetrix import JOVBaseNode, Lexicon, deep_merge

from Jovimetrix.sup.util import EnumConvertType, parse_param

from Jovimetrix.sup.midi import MIDIMessage, MIDINoteOnFilter, MIDIServerThread,\
      midi_device_names

# ==============================================================================

JOV_CATEGORY = "DEVICE"

# ==============================================================================

class MIDIMessageNode(JOVBaseNode):
    NAME = "MIDI MESSAGE (JOV) 🎛️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.VALUE, Lexicon.NORMALIZE, )
    SORT = 10
    DESCRIPTION = """
Processes MIDI messages received from an external MIDI controller or device. It accepts MIDI messages as input and returns various attributes of the MIDI message, including whether the message is valid, the MIDI channel, control number, note number, value, and normalized value. This node is useful for integrating MIDI control into creative projects, allowing users to respond to MIDI input in real-time.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.MIDI: ('JMIDIMSG', {"default": None})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[object, bool, int, int, int, float, float]:
        message: MIDIMessage = parse_param(kw, Lexicon.MIDI, EnumConvertType.ANY, [None])
        results = []
        pbar = ProgressBar(len(message))
        for idx, message in enumerate(message):
            if message is None:
                results.append([False, -1, -1, -1, -1, -1])
            else:
                results.append([message, *message.flat])
            pbar.update_absolute(idx)
        return list(zip(*results))

class MIDIReaderNode(JOVBaseNode):
    NAME = "MIDI READER (JOV) 🎹"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT',)
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.VALUE, Lexicon.NORMALIZE,)
    SORT = 5
    DEVICES = midi_device_names()
    DESCRIPTION = """
Captures MIDI messages from an external MIDI device or controller. It monitors MIDI input and provides information about the received MIDI messages, including whether a note is being played, the MIDI channel, control number, note number, value, and a normalized value. This node is essential for integrating MIDI control into various applications, such as music production, live performances, and interactive installations.
"""

    CHANGED = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.DEVICE : (cls.DEVICES, {"default": cls.DEVICES[0] if len(cls.DEVICES) > 0 else None})
            }
        })
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        if cls.CHANGED:
            cls.CHANGED = False
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
        MIDIReaderNode.CHANGED = True
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
        return msg, self.__note_on, self.__channel, self.__control, self.__note, self.__value, normalize,

class MIDIFilterNode(JOVBaseNode):
    NAME = "MIDI FILTER (JOV) ✳️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("JMIDIMSG", "BOOLEAN", )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.TRIGGER,)
    SORT = 20
    EPSILON = 1e-6
    DESCRIPTION = """
Provides advanced filtering capabilities for MIDI messages based on various criteria, including MIDI mode (such as note on or note off), MIDI channel, control number, note number, value, and normalized value. It allows you to filter out unwanted MIDI events and selectively process only the desired ones. This node offers flexibility in MIDI data processing, enabling precise control over which MIDI messages are passed through for further processing.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.MIDI: ('JMIDIMSG', {"default": None}),
                Lexicon.ON: (MIDINoteOnFilter._member_names_, {"default": MIDINoteOnFilter.IGNORE.name}),
                Lexicon.CHANNEL: ("STRING", {"default": ""}),
                Lexicon.CONTROL: ("STRING", {"default": ""}),
                Lexicon.NOTE: ("STRING", {"default": ""}),
                Lexicon.VALUE: ("STRING", {"default": ""}),
                Lexicon.NORMALIZE: ("STRING", {"default": ""}),
            }
        })
        return Lexicon._parse(d, cls)

    def __filter(self, data:int, value:str) -> bool:
        """Parse strings with number ranges into number ranges.
            1, 5-10, 2
        Would check == 1, == 2 and 5 <= x <= 10
        """
        value = value.strip()
        if value == "" or len(value) == 0 or value is None:
            return True
        ranges = value.split(',')
        for item in ranges:
            item = item.strip()
            if '-' in item:
                try:
                    a, b = map(float, item.split('-'))
                    if a <= data <= b:
                        return True
                except ValueError:
                    pass
                except Exception as e:
                    logger.error(e)
            else:
                try:
                    if isclose(data, float(item)):
                        return True
                except ValueError:
                    pass
                except Exception as e:
                    logger.error(e)
        return False

    def run(self, **kw) -> Tuple[bool]:
        message: MIDIMessage = kw.get(Lexicon.MIDI, None)
        note_on: str = parse_param(kw, Lexicon.ON, EnumConvertType.STRING, MIDINoteOnFilter.IGNORE.name)[0]
        chan: str = parse_param(kw, Lexicon.CHANNEL, EnumConvertType.STRING, "")[0]
        ctrl: str = parse_param(kw, Lexicon.CONTROL, EnumConvertType.STRING, "")[0]
        note: str = parse_param(kw, Lexicon.NOTE, EnumConvertType.STRING, "")[0]
        value: str = parse_param(kw, Lexicon.VALUE, EnumConvertType.STRING, "")[0]
        normal: str = parse_param(kw, Lexicon.NORMALIZE, EnumConvertType.STRING, "")[0]

        note_on = MIDINoteOnFilter[note_on]
        if note_on != MIDINoteOnFilter.IGNORE:
            if note_on == MIDINoteOnFilter.NOTE_ON and message.note_on != True:
                return message, False,
            if note_on == MIDINoteOnFilter.NOTE_OFF and message.note_on != False:
                return message, False,
        elif self.__filter(message.channel, chan) == False:
            return message, False,
        elif self.__filter(message.control, ctrl) == False:
            return message, False,
        elif self.__filter(message.note, note) == False:
            return message, False,
        elif self.__filter(message.value, value) == False:
            return message, False,
        elif self.__filter(message.normal, normal) == False:
            return message, False,
        return message, True,

class MIDIFilterEZNode(JOVBaseNode):
    NAME = "MIDI FILTER EZ (JOV) ❇️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("JMIDIMSG", "BOOLEAN", )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.TRIGGER,)
    SORT = 25
    DESCRIPTION = """
Filter MIDI messages based on various criteria, including MIDI mode (such as note on or note off), MIDI channel, control number, note number, value, and normalized value. This node is useful for processing MIDI input and selectively passing through only the desired messages. It helps simplify MIDI data handling by allowing you to focus on specific types of MIDI events.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.MIDI: ('JMIDIMSG', {"default": None}),
                Lexicon.MODE: (MIDINoteOnFilter._member_names_, {"default": MIDINoteOnFilter.IGNORE.name}),
                Lexicon.CHANNEL: ("INT", {"default": -1, "mij": -1, "maj": 127}),
                Lexicon.CONTROL: ("INT", {"default": -1, "mij": -1, "maj": 127}),
                Lexicon.NOTE: ("INT", {"default": -1, "mij": -1, "maj": 127}),
                Lexicon.VALUE: ("INT", {"default": -1, "mij": -1, "maj": 127}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[MIDIMessage, bool]:

        message: MIDIMessage = parse_param(kw, Lexicon.MIDI, EnumConvertType.ANY, [None])[0]
        note_on = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, MIDINoteOnFilter.IGNORE.name)[0]
        chan = parse_param(kw, Lexicon.CHANNEL, EnumConvertType.INT, -1)[0]
        ctrl = parse_param(kw, Lexicon.CONTROL, EnumConvertType.INT, -1)[0]
        note = parse_param(kw, Lexicon.NOTE, EnumConvertType.INT, -1)[0]
        value = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, -1)[0]

        note_on = MIDINoteOnFilter[note_on]
        if note_on != MIDINoteOnFilter.IGNORE:
            if note_on == MIDINoteOnFilter.NOTE_ON and message.note_on != True:
                return message, False,
            if note_on == MIDINoteOnFilter.NOTE_OFF and message.note_on != False:
                return message, False,
        elif chan > -1 and message.channel != chan:
            return message, False,
        elif ctrl > -1 and message.control != ctrl:
            return message, False,
        elif note > -1 and message.note != note:
            return message, False,
        elif value > -1 and message.value != value:
            return message, False,
        return message, True,
