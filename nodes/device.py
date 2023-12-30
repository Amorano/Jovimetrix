"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Device -- MIDI, WEBCAM

    type 0 (single track): all messages are saved in one track
    type 1 (synchronous): all tracks start at the same time
    type 2 (asynchronous): each track is independent of the others
"""

import time
import uuid
import threading
from math import isclose
from enum import Enum
from queue import Queue, Empty

import cv2
import torch
import numpy as np

from Jovimetrix import parse_tuple, parse_number, deep_merge_dict, tensor2cv, \
    cv2mask, cv2tensor, zip_longest_fill, \
    JOVBaseNode, JOVImageBaseNode, JOVImageInOutBaseNode, Logger, Lexicon, \
    EnumTupleType, IT_SCALEMODE, \
    MIN_IMAGE_SIZE, IT_PIXELS, IT_ORIENT, IT_CAM, IT_WHMODE, IT_REQUIRED, IT_INVERT

from Jovimetrix.sup.comp import image_grid, light_invert, geo_scalefit, \
    EnumInterpolation, EnumScaleMode, \
    IT_SAMPLE
from Jovimetrix.sup.stream import MediaStreamBase, MediaStreamDevice, StreamingServer, StreamManager

try:
    import mido
    from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
except:
    Logger.warn("MISSING MIDI SUPPORT")

def save_midi() -> None:
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('key_signature', key='Dm'))
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(120)))
    track.append(MetaMessage('time_signature', numerator=6, denominator=8))

    track.append(Message('program_change', program=12, time=10))
    track.append(Message('note_on', channel=2, note=60, velocity=64, time=1))
    track.append(Message('note_off', channel=2, note=60, velocity=100, time=2))

    track.append(MetaMessage('end_of_track'))
    mid.save('new_song.mid')

def load_midi(fn) -> None:
    mid = MidiFile(fn, clip=True)
    Logger.debug(mid)
    for msg in mid.tracks[0]:
        Logger.debug(msg)

class MIDIServerThread(threading.Thread):
    def __init__(self, q_in, device, callback, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__q_in = q_in
        # self.__q_out = q_out
        self.__device = device
        self.__callback = callback

    def __run(self) -> None:
        with mido.open_input(self.__device, callback=self.__callback) as inport:
            while True:
                try:
                    cmd = self.__q_in.get_nowait()
                    if (cmd):
                        self.__device = cmd
                        break
                except Empty as _:
                    time.sleep(0.01)
                except Exception as e:
                    Logger.debug(str(e))

    def run(self) -> None:
        while True:
            Logger.debug(f"started device loop {self.__device}")
            try:
                self.__run()
            except Exception as e:
                if self.__device is None:
                    try:
                        cmd = self.__q_in.get_nowait()
                        if (cmd):
                            self.__device = cmd
                            break
                    except Empty as _:
                        time.sleep(0.01)
                    except Exception as e:
                        Logger.debug(str(e))
                Logger.err(str(e))

# =============================================================================

class StreamReaderNode(JOVImageBaseNode):
    NAME = "STREAM READER (JOV) 📺"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )
    EMPTY = np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.float32)
    SORT = 50

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.URL: ("STRING", {"default": "0"}),
            Lexicon.FPS: ("INT", {"min": 1, "max": 60, "step": 1, "default": 30}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            Lexicon.WH: ("VEC2", {"default": (320, 240), "min": MIN_IMAGE_SIZE, "max": 8192, "step": 1, "label": [Lexicon.WIDTH, Lexicon.HEIGHT]})

        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_SCALEMODE, IT_SAMPLE, IT_INVERT, IT_ORIENT, IT_CAM)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self) -> None:
        self.__device:[MediaStreamBase|MediaStreamDevice] = None
        self.__url = ""
        self.__last = StreamReaderNode.EMPTY
        self.__capturing = 0

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        url = kw[Lexicon.URL]
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1])[0]
        width, height = parse_tuple(Lexicon.WH, kw)[0]

        if self.__capturing == 0 and (self.__device is None or url != self.__url): #or not self.__device.captured:
            self.__capturing = time.perf_counter()
            Logger.debug('capturing', self.__device, self.__capturing, self.__url, url)
            self.__url = url
            try:
                self.__device = StreamManager().capture(url, width, height)
            except Exception as e:
                Logger.err(str(e))

        if self.__capturing > 0:
            # timeout and try again?
            if time.perf_counter() - self.__capturing > 5000:
                Logger.err(f'timed out trying to access route or device {self.__url}')
                self.__capturing = 0
                self.__url = ""

        img = None
        if self.__device:
            self.__capturing = 0
            ret, img = self.__device.frame
            self.__last = img if img is not None else self.__last
            if ret:
                h, w = self.__last.shape[:2]
                if width != w or height != h:
                    self.__device.sizer(width, height)
                    mode = kw[Lexicon.MODE]
                    rs = kw[Lexicon.SAMPLE]
                    img = geo_scalefit(img, width, height, mode, EnumInterpolation[rs])

                orient = kw[Lexicon.ORIENT]
                if orient in ["FLIPX", "FLIPXY"]:
                    img = cv2.flip(img, 1)

                if orient in ["FLIPY", "FLIPXY"]:
                    img = cv2.flip(img, 0)

                if i != 0:
                    img = light_invert(img, i)

                if self.__device.fps != (fps := kw[Lexicon.FPS]):
                    self.__device.fps = fps

                if self.__device.zoom != kw[Lexicon.ZOOM]:
                    self.__device.zoom = kw[Lexicon.ZOOM]

                if kw[Lexicon.WAIT]:
                    self.__device.pause()
                else:
                    self.__device.play()

        if img is None:
            self.__last = img = StreamReaderNode.EMPTY
        return ( cv2tensor(img), cv2mask(img) )

class StreamWriterNode(JOVImageInOutBaseNode):
    NAME = "STREAM WRITER (JOV) 🎞️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    SORT = 70
    OUT_MAP = {}

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.ROUTE: ("STRING", {"default": "/stream"}),
                Lexicon.WH: ("VEC2", {"default": (640, 480), "min": MIN_IMAGE_SIZE, "max": 8192, "step": 1, "label": [Lexicon.WIDTH, Lexicon.HEIGHT]})
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_SCALEMODE, IT_SAMPLE, IT_INVERT)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

        route = kw.get(Lexicon.ROUTE, ["/stream"])
        # width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)

        if (StreamManager().capture(route, static=True)) is None:
            Logger.err(f"stream failed {route}")

        # if device.size[0] != width or device.size[1] != height:
        #     device.size = (width, height)

        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__ss = StreamingServer()
        self.__route = ""
        self.__unique = uuid.uuid4()
        self.__device = None
        StreamWriterNode.OUT_MAP[self.__unique] = None

    def run(self, **kw) -> tuple[torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        route = kw[Lexicon.ROUTE]
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        mode = kw[Lexicon.MODE]
        sample = kw[Lexicon.SAMPLE]
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        stride = len(pixels)
        grid = int(np.sqrt(stride))
        if grid * grid < stride:
            grid += 1
        out = []
        for img, r, wihi, mode, rs, i in zip_longest_fill(pixels, route, wihi, mode, sample, i):
            w, h = wihi
            img = tensor2cv(img) if img is not None else np.zeros((h, w, 3), dtype=np.uint8)
            if r != self.__route:
                # close old, if any
                if self.__device:
                    self.__device.release()

                # startup server
                self.__device = StreamManager().capture(self.__unique, static=True)
                self.__ss.endpointAdd(r, self.__device)
                self.__route = r
                Logger.debug(self, "START", r)

            sw, sh = w // stride, h // stride
            if self.__device:
                try:
                    img = geo_scalefit(img, sw, sh, EnumScaleMode.NONE)
                except Exception as e:
                    Logger.err(str(e))

            if i != 0:
                img = light_invert(img, i)
            out.append(img)

        if len(out) > 1:
            img = image_grid(out, w, h)
        else:
            img = out[0]
        img = geo_scalefit(img, w, h, mode, EnumInterpolation[rs])
        # self.__device.post(img)
        return (cv2tensor(img), cv2mask(img),)

class MIDIMessage:
    """Snap shot of a message from Midi device."""
    def __init__(self, note_on, channel, control, note, value) -> None:
        self.note_on = note_on
        self.channel = channel
        self.control = control
        self.note = note
        self.value = value
        self.normal = value / 127.

    @property
    def flat(self) -> tuple[bool, int, int, int, float, float]:
        return (self.note_on, self.channel, self.control, self.note, self.value, self.normal,)

    def __str__(self) -> str:
        return f"{self.note_on}, {self.channel}, {self.control}, {self.note}, {self.value}, {self.normal}"

class MIDIMessageNode(JOVBaseNode):
    NAME = "MIDI MESSAGE (JOV) 🎛️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = "Expands a MIDI message into its values."
    OUTPUT_IS_LIST = (False, False, False, False, False, False, False,)
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.AMT, Lexicon.NORMALIZE, )
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.MIDI: ('JMIDIMSG', {"default": None})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[object, bool, int, int, int, float, float]:
        if (message := kw.get(Lexicon.MIDI, None)) is None:
            return message, False, -1, -1, -1, -1, -1
        return message, *message.flat

class MIDIReaderNode(JOVBaseNode):
    NAME = "MIDI READER (JOV) 🎹"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = "Reads input from a midi device"
    OUTPUT_IS_LIST = (False, False, False, False, False, False, False)
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT',)
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.AMT, Lexicon.NORMALIZE,)
    SORT = 5
    DEVICES = mido.get_input_names()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data = mido.get_input_names()
        d = {"optional": {
            Lexicon.DEVICE : (cls.DEVICES, {"default": cls.DEVICES[0] if len(cls.DEVICES) > 0 else None})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def __init__(self) -> None:
        self.__q_in = Queue()
        self.__q_out = Queue()
        self.__device = None
        self.__note = 0
        self.__note_on = False
        self.__channel = 0
        self.__control = 0
        self.__value = 0
        MIDIReaderNode.DEVICES = mido.get_input_names()
        self.__SERVER = MIDIServerThread(self.__q_in, self.__device, self.__process, daemon=True)
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
                # note=59 velocity=0 time=0
            case "note_off":
                self.__note = data.note
                self.__value = data.velocity
                # note=59 velocity=0 time=0

        # Logger.spam(self.__note_on, self.__channel, self.__control, self.__note, self.__value)

    def run(self, **kw) -> tuple[bool, int, int, int]:
        device = kw.get(Lexicon.DEVICE, None)

        if device != self.__device:
            self.__q_in.put(device)
            self.__device = device

        normalize = self.__value / 127.
        Logger.spam(self.__note_on, self.__channel, self.__control, self.__note, self.__value, normalize)
        msg = MIDIMessage(self.__note_on, self.__channel, self.__control, self.__note, self.__value)
        return (msg, self.__note_on, self.__channel, self.__control, self.__note, self.__value, normalize,  )

class MIDINoteOnFilter(Enum):
    FALSE = 0
    TRUE = 1
    IGNORE = -1

class MIDIFilterEZNode(JOVBaseNode):
    NAME = "MIDI FILTER EZ ❇️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = "Filter MIDI messages by channel, message type or value."
    OUTPUT_IS_LIST = (False, False, )
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.TRIGGER,)
    SORT = 25
    # EPSILON = 1 / 128.

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.MIDI: ('JMIDIMSG', {"default": None}),
            Lexicon.MODE: (MIDINoteOnFilter._member_names_, {"default": MIDINoteOnFilter.IGNORE.name}),
            Lexicon.CHANNEL: ("INT", {"default": -1, "min": -1, "max": 127, "step": 1}),
            Lexicon.CONTROL: ("INT", {"default": -1, "min": -1, "max": 127, "step": 1}),
            Lexicon.NOTE: ("INT", {"default": -1, "min": -1, "max": 127, "step": 1}),
            Lexicon.VALUE: ("INT", {"default": -1, "min": -1, "max": 127, "step": 1}),
            Lexicon.NORMALIZE: ("FLOAT", {"default": -1, "min": -1, "max": 1, "step": 0.01})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, **kw) -> tuple[bool]:
        message = kw.get(Lexicon.MIDI, None)
        if message is None:
            Logger.debug('no midi message. connected?')
            return (message, False, )

        # empty values mean pass-thru (no filter)
        if (val := kw[Lexicon.MODE]) != MIDINoteOnFilter.IGNORE:
            if val == "TRUE" and message.note_on != True:
                return (message, False, )
            if val == "FALSE" and message.note_on != False:
                return (message, False, )
        if (val := kw[Lexicon.CHANNEL]) != -1 and val != message.channel:
            return (message, False, )
        if (val := kw[Lexicon.CONTROL]) != -1 and val != message.control:
            return (message, False, )
        if (val := kw[Lexicon.NOTE]) != -1 and val != message.note:
            return (message, False, )
        if (val := kw[Lexicon.VALUE]) != -1 and val != message.value:
            return (message, False, )
        if (val := kw[Lexicon.NORMALIZE]) != -1 and isclose(val, message.normal):
            return (message, False, )
        return (message, True, )

class MIDIFilterNode(JOVBaseNode):
    NAME = "MIDI FILTER ✳️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = "Filter MIDI messages by channel, message type or value."
    OUTPUT_IS_LIST = (False, False, )
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.TRIGGER,)
    SORT = 20
    EPSILON = 1e-6

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.MIDI: ('JMIDIMSG', {"default": None}),
            Lexicon.ON: (MIDINoteOnFilter._member_names_, {"default": MIDINoteOnFilter.IGNORE.name}),
            Lexicon.CHANNEL: ("STRING", {"default": ""}),
            Lexicon.CONTROL: ("STRING", {"default": ""}),
            Lexicon.NOTE: ("STRING", {"default": ""}),
            Lexicon.VALUE: ("STRING", {"default": ""}),
            Lexicon.NORMALIZE: ("STRING", {"default": ""})
        }}
        return deep_merge_dict(IT_REQUIRED, d)

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
            Logger.spam(str(e))

        for line in data.split(','):
            if len(a_range := line.split('-')) > 1:
                try:
                    a, b = a_range[:2]
                    if float(a) <= value <= float(b):
                        return True
                except Exception as e:
                    Logger.spam(str(e))

            try:
                if isclose(value, float(line)):
                    return True
            except Exception as e:
                Logger.spam(str(e))
        return False

    def run(self, **kw) -> tuple[bool]:
        message = kw.get(Lexicon.MIDI, None)
        if message is None:
            Logger.debug('no midi message. connected?')
            return (message, False, )

        # empty values mean pass-thru (no filter)
        if (val := kw[Lexicon.ON]) != MIDINoteOnFilter.IGNORE:
            if val == "TRUE" and message.note_on != True:
                return (message, False, )
            if val == "FALSE" and message.note_on != False:
                return (message, False, )
        if self.__filter(kw[Lexicon.CHANNEL], message.channel) == False:
            return (message, False, )
        if self.__filter(kw[Lexicon.CONTROL], message.control) == False:
            return (message, False, )
        if self.__filter(kw[Lexicon.NOTE], message.note) == False:
            return (message, False, )
        if self.__filter(kw[Lexicon.VALUE], message.value) == False:
            return (message, False, )
        if self.__filter(kw[Lexicon.NORMALIZE], message.normal) == False:
            return (message, False, )
        return (message, True, )

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":

    def process(data) -> None:
        channel = data.channel
        note = 0
        control = 0
        note_on = False
        match data.type:
            case "control_change":
                # control=8 value=14 time=0
                control = data.control
                value = data.value
            case "note_on":
                note = data.note
                note_on = True
                value = data.velocity
                # note=59 velocity=0 time=0
            case "note_off":
                note = data.note
                value = data.velocity
                # note=59 velocity=0 time=0

        value /= 127.
        Logger.debug(note_on, channel, control, note, value)

    device= mido.get_input_names()[0]
    Logger.debug(device)
    q_in = Queue()
    q_out = Queue()
    server = MIDIServerThread(q_in, device, process, daemon=True)
    server.start()
    while True:
        time.sleep(0.01)
