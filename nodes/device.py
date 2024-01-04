"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Device -- MIDI, WEBCAM

    type 0 (single track): all messages are saved in one track
    type 1 (synchronous): all tracks start at the same time
    type 2 (asynchronous): each track is independent of the others
"""

import time
import uuid
from math import isclose
from queue import Queue
from enum import Enum

import cv2
import torch
import numpy as np

from Jovimetrix import parse_tuple, parse_number, deep_merge_dict, \
    JOVBaseNode, JOVImageBaseNode, Logger, Lexicon, EnumTupleType, \
    MIN_IMAGE_SIZE, IT_PIXELS, IT_CAM, IT_REQUIRED, IT_INVERT

from Jovimetrix.sup.comp import light_invert, geo_scalefit, \
    EnumInterpolation, EnumScaleMode

from Jovimetrix.sup.stream import camera_list, StreamingServer, StreamManager

from Jovimetrix.sup.midi import midi_device_names, \
    MIDIMessage, MIDINoteOnFilter, MIDIServerThread

from Jovimetrix.sup.image import channel_count, tensor2cv, cv2mask, cv2tensor, IT_SAMPLE, IT_SCALEMODE

# =============================================================================

class EnumCanvasOrientation(Enum):
    NORMAL = 0
    FLIPX = 1
    FLIPY = 2
    FLIPXY = 3

# =============================================================================

IT_ORIENT = {"optional": {
    Lexicon.ORIENT: (EnumCanvasOrientation._member_names_, {"default": EnumCanvasOrientation.NORMAL.name}),
}}

# =============================================================================

class StreamReaderNode(JOVImageBaseNode):
    NAME = "STREAM READER (JOV) 📺"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (True, True, )
    SORT = 50
    CAMERA_LIST = None

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        if cls.CAMERA_LIST is None:
            cls.CAMERA_LIST = camera_list()
        default = cls.CAMERA_LIST[0] if len(cls.CAMERA_LIST) > 0 else "NONE"
        d = {"optional": {
            Lexicon.URL: ("STRING", {"default": ""}),
            Lexicon.CAMERA: (cls.CAMERA_LIST, {"default": default}),
            Lexicon.FPS: ("INT", {"min": 1, "max": 60, "default": 30}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            Lexicon.WH: ("VEC2", {"default": (320, 240), "min": MIN_IMAGE_SIZE, "max": 8192, "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.BATCH: ("VEC2", {"default": (1, 30), "min": 1, "step": 1, "label": ["BATCH", ""]}),
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_SCALEMODE, IT_SAMPLE, IT_INVERT, IT_ORIENT, IT_CAM)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self) -> None:
        self.__device = None
        self.__url = ""
        self.__capturing = 0

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        url = kw.get(Lexicon.URL, "")
        if url == "":
            url = kw.get(Lexicon.CAMERA, None)
            try:
                _ = int(url)
                url = str(url)
            except: url = ""

        width, height = parse_tuple(Lexicon.WH, kw, default=(320, 240,), clip_min=MIN_IMAGE_SIZE)[0]

        if self.__capturing == 0 and (self.__device is None or url != self.__url):
            self.__capturing = time.perf_counter()
            # Logger.debug('capturing', self.__capturing, self.__url, url)
            self.__url = url
            try:
                self.__device = StreamManager().capture(url, width, height)
            except Exception as e:
                Logger.err(str(e))

        if self.__capturing > 0:
            # timeout and try again?
            if time.perf_counter() - self.__capturing > 3000:
                Logger.err(f'timed out trying to access route or device {self.__url}')
                self.__capturing = 0
                self.__url = ""

        images = []
        masks = []
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1])[0]
        if self.__device:
            self.__capturing = 0

            fps = kw.get(Lexicon.FPS, 30)
            if self.__device.fps != fps:
                self.__device.fps = fps

            zoom = kw.get(Lexicon.ZOOM, 0)
            if self.__device.zoom != zoom:
                self.__device.zoom = zoom

            if kw.get(Lexicon.WAIT, False):
                self.__device.pause()
            else:
                self.__device.play()

            mode = kw.get(Lexicon.MODE, EnumScaleMode.NONE)
            mode = EnumScaleMode[mode]
            rs = kw.get(Lexicon.SAMPLE, EnumInterpolation.LANCZOS4)
            rs = EnumInterpolation[rs]
            orient = kw.get(Lexicon.ORIENT, EnumCanvasOrientation.NORMAL)
            orient = EnumCanvasOrientation[orient]
            batch_size, rate = parse_tuple(Lexicon.BATCH, kw, default=(1, 30), clip_min=1)[0]
            rate = 1. / rate
            for idx in range(batch_size):
                mask = None
                ret, img = self.__device.frame
                if img is None:
                    img = np.zeros((height, width, 3), dtype=np.uint8)

                if ret:
                    cc, _, w, h = channel_count(img)
                    # drop the alpha?
                    if cc == 4:
                        mask = img[:, :, 3]
                        img = img[:, :, :3]

                    if width != w or height != h:
                        self.__device.sizer(width, height)
                        img = geo_scalefit(img, width, height, mode, rs)

                    if orient in [EnumCanvasOrientation.FLIPX, EnumCanvasOrientation.FLIPXY]:
                        img = cv2.flip(img, 1)

                    if orient in [EnumCanvasOrientation.FLIPY, EnumCanvasOrientation.FLIPXY]:
                        img = cv2.flip(img, 0)

                    if i != 0:
                        img = light_invert(img, i)

                images.append(cv2tensor(img))
                if mask is None:
                    mask = np.full((height, width), 255, dtype=np.uint8)
                masks.append(cv2mask(mask))

                if batch_size > 1:
                    time.sleep(rate)
                    # Logger.debug(idx, rate)

        if len(images) == 0:
            images = [torch.zeros((height, width, 3), dtype=torch.uint8, device="cpu")]
            masks = [torch.full((height, width), 1, dtype=torch.uint8, device="cpu")]

        return (torch.stack(images), torch.stack(masks))

class StreamWriterNode(JOVBaseNode):
    NAME = "STREAM WRITER (JOV) 🎞️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    OUTPUT_NODE = True
    SORT = 70
    OUT_MAP = {}

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.ROUTE: ("STRING", {"default": "/stream"}),
                Lexicon.WH: ("VEC2", {"default": (640, 480), "min": MIN_IMAGE_SIZE, "max": 8192, "step": 1, "label": [Lexicon.W, Lexicon.H]})
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_SCALEMODE, IT_SAMPLE, IT_INVERT)

    #@classmethod
    #def IS_CHANGED(cls, **kw) -> float:
    #    return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__route = ""
        self.__unique = uuid.uuid4()
        self.__device = None
        self.__starting = False

    def run(self, **kw) -> tuple[torch.Tensor]:
        if self.__starting:
            return

        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        w, h = wihi
        img = kw.get(Lexicon.PIXEL, None)
        img = tensor2cv(img) if img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        route = kw.get(Lexicon.ROUTE, "/stream")
        if route != self.__route:
            self.__starting = True
            # close old, if any
            if self.__device:
                self.__device.release()
            # startup server
            self.__device = StreamManager().capture(self.__unique, static=True)
            StreamingServer().endpointAdd(route, self.__device)
            StreamWriterNode.OUT_MAP[route] = self.__device
            self.__route = route
            # Logger.debug(self, "START", route)

        self.__starting = False
        if self.__device is not None:
            mode = kw.get(Lexicon.MODE, EnumScaleMode.NONE)
            mode = EnumScaleMode[mode]
            rs = kw.get(Lexicon.SAMPLE, EnumInterpolation.LANCZOS4)
            rs = EnumInterpolation[rs]
            i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)[0]
            img = geo_scalefit(img, w, h, EnumScaleMode.NONE)
            if i != 0:
                img = light_invert(img, i)
            img = geo_scalefit(img, w, h, mode, rs)
            self.__device.image = img
        return ()

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
    DEVICES = midi_device_names()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
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
        MIDIReaderNode.DEVICES = midi_device_names()
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
            Logger.warn('no midi message. connected?')
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
            Logger.warn('no midi message. connected?')
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
