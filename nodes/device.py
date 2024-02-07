"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Device -- MIDI, WEBCAM

    type 0 (single track): all messages are saved in one track
    type 1 (synchronous): all tracks start at the same time
    type 2 (asynchronous): each track is independent of the others
"""

import sys
import time
import uuid
from math import isclose
from queue import Queue
from enum import Enum

import cv2
import torch
import numpy as np
from loguru import logger

import comfy

from Jovimetrix import JOVBaseNode, JOVImageMultiple, \
    IT_PIXEL, IT_INVERT, IT_REQUIRED, MIN_IMAGE_SIZE

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import deep_merge_dict, parse_tuple, parse_number, \
    EnumTupleType

from Jovimetrix.sup.stream import camera_list, monitor_list, window_list, \
    monitor_capture, window_capture, \
    StreamingServer, StreamManager

from Jovimetrix.sup.midi import midi_device_names, \
    MIDIMessage, MIDINoteOnFilter, MIDIServerThread

from Jovimetrix.sup.image import channel_count, cv2mask, tensor2cv, cv2tensor, \
    image_scalefit, image_invert, \
    EnumInterpolation, EnumScaleMode, \
    IT_WHMODE, IT_SAMPLE, IT_SCALEMODE

# =============================================================================

class EnumCanvasOrientation(Enum):
    NORMAL = 0
    FLIPX = 1
    FLIPY = 2
    FLIPXY = 3

# =============================================================================

class StreamReaderNode(JOVImageMultiple):
    NAME = "STREAM READER (JOV) 📺"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    SORT = 50
    CAMERAS = None

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        if cls.CAMERAS is None:
            cls.CAMERAS = [f"{i} - {v['w']}x{v['h']}" for i, v in enumerate(camera_list().values())]
        camera_default = cls.CAMERAS[0] if len(cls.CAMERAS) else "NONE"

        monitors = monitor_list()
        monitors.pop(0)
        monitor = [f"{i} - {v['width']}x{v['height']}" for i, v in enumerate(monitors.values())]

        window = []
        if sys.platform.startswith('win'):
            window = [f"{v} - {k}" for k, v in window_list().items()]
        window_default = window[0] if len(window) else "NONE"

        d = {"optional": {
            Lexicon.SOURCE: (["URL", "CAMERA", "MONITOR", "WINDOW"], {"default": "URL"}),

            Lexicon.URL: ("STRING", {"default": "", "dynamicPrompts": False}),
            Lexicon.CAMERA: (cls.CAMERAS, {"default": camera_default}),
            Lexicon.MONITOR: (monitor, {"default": monitor[0]}),
            Lexicon.WINDOW: (window, {"default": window_default}),
            Lexicon.DPI: ("BOOLEAN", {"default": True}),
            Lexicon.BBOX: ("VEC4", {"default": (0, 0, 1, 1), "min": 0, "max": 1, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
            Lexicon.FPS: ("INT", {"min": 1, "max": 60, "default": 30}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            Lexicon.BATCH: ("VEC2", {"default": (1, 30), "min": 1, "step": 1, "label": ["COUNT", "FPS"]}),
        }}

        e = {"optional": {
            Lexicon.ORIENT: (EnumCanvasOrientation._member_names_, {"default": EnumCanvasOrientation.NORMAL.name}),
            Lexicon.ZOOM: ("FLOAT", {"min": 0, "max": 1, "step": 0.005, "default": 0}),
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WHMODE, IT_SAMPLE, e, IT_INVERT)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__device = None
        self.__url = ""
        self.__capturing = 0
        self.__last = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu")
        self.__last_mask = torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=torch.uint8, device="cpu")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        masks = []
        batch_size, rate = parse_tuple(Lexicon.BATCH, kw, default=(1, 30), clip_min=1)[0]
        pbar = comfy.utils.ProgressBar(batch_size)
        rate = 1. / rate
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))[0]
        wait = kw.get(Lexicon.WAIT, False)
        mode = kw.get(Lexicon.MODE, EnumScaleMode.NONE)
        mode = EnumScaleMode[mode]
        sample = kw.get(Lexicon.SAMPLE, EnumInterpolation.LANCZOS4)
        sample = EnumInterpolation[sample]
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1])[0]
        source = kw.get(Lexicon.SOURCE, "URL")
        empty = (torch.stack([self.__last]), torch.stack([self.__last_mask]), )
        match source:
            case "MONITOR":
                if wait:
                    return empty
                which = kw.get(Lexicon.MONITOR, "0")
                which = int(which.split('-')[0].strip()) + 1
                for idx in range(batch_size):
                    img = monitor_capture(which)
                    if img is None:
                        img = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        if i != 0:
                            img = image_invert(img, i)
                        img = image_scalefit(img, width, height, mode=mode, sample=sample)

                    cc, w, h = channel_count(img)[:3]
                    if cc == 4:
                        mask = img[:, :, 3]
                        img = img[:, :, :3]
                    else:
                        mask = np.ones((h, w), dtype=np.uint8) * 255

                    images.append(cv2tensor(img))
                    masks.append(cv2mask(mask))

                    if batch_size > 1:
                        pbar.update_absolute(idx)
                        time.sleep(rate)

            case "WINDOW":
                if wait:
                    return empty
                if (which := kw.get(Lexicon.WINDOW, "NONE")) != "NONE":
                    which = int(which.split('-')[-1].strip())
                    dpi = kw.get(Lexicon.DPI, True)
                    for idx in range(batch_size):
                        img = window_capture(which, dpi=dpi)
                        if img is not None:
                            if i != 0:
                                img = image_invert(img, i)
                            img = image_scalefit(img, width, height, mode=mode, sample=sample)
                        else:
                            img = np.zeros((height, width, 3), dtype=np.uint8)

                        cc, w, h = channel_count(img)[:3]
                        if cc == 4:
                            mask = img[:, :, 3]
                            img = img[:, :, :3]
                        else:
                            mask = np.ones((h, w), dtype=np.uint8) * 255

                        images.append(cv2tensor(img))
                        masks.append(cv2mask(mask))

                        if batch_size > 1:
                            pbar.update_absolute(idx)
                            time.sleep(rate)

            case "URL" | "CAMERA":
                url = kw.get(Lexicon.URL, "")
                if source == "CAMERA":
                    url = kw.get(Lexicon.CAMERA, "")
                    url = url.split('-')[0].strip()
                    try:
                        _ = int(url)
                        url = str(url)
                    except: url = ""

                if self.__capturing == 0 and (self.__device is None or url != self.__url):
                    self.__capturing = time.perf_counter()
                    self.__url = url
                    try:
                        self.__device = StreamManager().capture(url)
                    except Exception as e:
                        logger.error(str(e))

                if self.__capturing > 0:
                    # timeout and try again?
                    if time.perf_counter() - self.__capturing > 3000:
                        logger.error(f'timed out {self.__url}')
                        self.__capturing = 0
                        self.__url = ""

                if self.__device:
                    self.__capturing = 0

                    fps = kw.get(Lexicon.FPS, 30)
                    if self.__device.fps != fps:
                        self.__device.fps = fps

                    # only cameras have a zoom
                    try:
                        self.__device.zoom = kw.get(Lexicon.ZOOM, 0)
                    except Exception as e:
                        logger.error(e)

                    if wait:
                        self.__device.pause()
                    else:
                        self.__device.play()

                    orient = kw.get(Lexicon.ORIENT, EnumCanvasOrientation.NORMAL)
                    orient = EnumCanvasOrientation[orient]
                    for idx in range(batch_size):
                        _, img = self.__device.frame
                        if img is None:
                            img = np.zeros((height, width, 3), dtype=np.uint8)
                        else:
                            if orient in [EnumCanvasOrientation.FLIPX, EnumCanvasOrientation.FLIPXY]:
                                img = cv2.flip(img, 1)

                            if orient in [EnumCanvasOrientation.FLIPY, EnumCanvasOrientation.FLIPXY]:
                                img = cv2.flip(img, 0)

                            if i != 0:
                                img = image_invert(img, i)
                            img = image_scalefit(img, width, height, mode=mode, sample=sample)

                        cc, w, h = channel_count(img)[:3]
                        if cc == 4:
                            mask = img[:, :, 3]
                            img = img[:, :, :3]
                        else:
                            mask = np.ones((h, w), dtype=np.uint8) * 255

                        images.append(cv2tensor(img))
                        masks.append(cv2mask(mask))

                        pbar.update_absolute(idx)
                        if batch_size > 1:
                            time.sleep(rate)

        if len(images) == 0:
            return empty

        self.__last = images[-1]
        self.__last_mask = masks[-1]
        return images, masks,

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
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL, d, IT_SCALEMODE, IT_SAMPLE, IT_INVERT)

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
            # logger.debug("{} {}", "START", route)

        self.__starting = False
        if self.__device is not None:
            mode = kw.get(Lexicon.MODE, EnumScaleMode.NONE)
            mode = EnumScaleMode[mode]
            rs = kw.get(Lexicon.SAMPLE, EnumInterpolation.LANCZOS4)
            rs = EnumInterpolation[rs]
            i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)[0]
            img = image_scalefit(img, w, h, mode=EnumScaleMode.NONE)
            if i != 0:
                img = image_invert(img, i)
            img = image_scalefit(img, w, h, mode=mode, sample=rs)
            self.__device.image = img
        return ()

class MIDIMessageNode(JOVBaseNode):
    NAME = "MIDI MESSAGE (JOV) 🎛️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = "Expands a MIDI message into its values."
    OUTPUT_IS_LIST = (False, False, False, False, False, False, False,)
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.VALUE, Lexicon.NORMALIZE, )
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
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.VALUE, Lexicon.NORMALIZE,)
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

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
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

        # logger.debug("{} {} {} {} {}", self.__note_on, self.__channel, self.__control, self.__note, self.__value)

    def run(self, **kw) -> tuple[bool, int, int, int]:
        device = kw.get(Lexicon.DEVICE, None)

        if device != self.__device:
            self.__q_in.put(device)
            self.__device = device

        normalize = self.__value / 127.
        logger.debug("{} {} {} {} {} {}", self.__note_on, self.__channel, self.__control, self.__note, self.__value, normalize)
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
            logger.warning('no midi message. connected?')
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

    def run(self, **kw) -> tuple[bool]:
        message = kw.get(Lexicon.MIDI, None)
        if message is None:
            logger.warning('no midi message. connected?')
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
