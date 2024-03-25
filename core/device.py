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
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import load_help, JOVBaseNode, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import EnumConvertType, parse_parameter
from Jovimetrix.sup.stream import camera_list, monitor_list, window_list, \
    monitor_capture, window_capture, JOV_SPOUT, \
    StreamingServer, StreamManager, MediaStreamDevice

if JOV_SPOUT:
    from Jovimetrix.sup.stream import SpoutSender, MediaStreamSpout

from Jovimetrix.sup.midi import midi_device_names, \
    MIDIMessage, MIDINoteOnFilter, MIDIServerThread

from Jovimetrix.sup.image import EnumImageType, channel_solid, \
    cv2tensor, cv2tensor_full, image_convert, pixel_eval, \
    tensor2cv, image_scalefit, \
    EnumInterpolation, EnumScaleMode, MIN_IMAGE_SIZE

from Jovimetrix.sup.audio import AudioDevice

# =============================================================================

JOV_CATEGORY = "DEVICE"

class EnumCanvasOrientation(Enum):
    NORMAL = 0
    FLIPX = 1
    FLIPY = 2
    FLIPXY = 3

class EnumStreamType(Enum):
    URL = 10
    CAMERA = 20
    MONITOR = 30
    WINDOW = 40
    SPOUT = 50

# =============================================================================

class StreamReaderNode(JOVBaseNode):
    NAME = "STREAM READER (JOV) 📺"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-stream-reader"
    DESC = "Connect system media devices and remote streams into ComfyUI workflows."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    INPUT_IS_LIST = False
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
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

        names = EnumStreamType._member_names_
        if not JOV_SPOUT:
            names.pop()

        d = {"required": {},
             "optional": {
            Lexicon.SOURCE: (names, {"default": EnumStreamType.URL.name}),
            Lexicon.URL: ("STRING", {"default": "", "dynamicPrompts": False}),
            Lexicon.CAMERA: (cls.CAMERAS, {"default": camera_default}),
            Lexicon.MONITOR: (monitor, {"default": monitor[0]}),
            Lexicon.WINDOW: (window, {"default": window_default}),
            Lexicon.DPI: ("BOOLEAN", {"default": True}),
            Lexicon.BBOX: ("VEC4", {"default": (0, 0, 1, 1), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
            Lexicon.FPS: ("INT", {"min": 1, "max": 60, "default": 30}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            Lexicon.BATCH: ("VEC2", {"default": (1, 30), "step": 1, "label": ["COUNT", "FPS"]}),
            Lexicon.ORIENT: (EnumCanvasOrientation._member_names_, {"default": EnumCanvasOrientation.NORMAL.name}),
            Lexicon.ZOOM: ("FLOAT", {"min": 0, "max": 1, "step": 0.005, "default": 0}),
            Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__device = None
        self.__deviceType = None
        self.__url = ""
        self.__capturing = 0
        a = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=torch.uint8, device="cpu")
        e = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu")
        m = torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 1), dtype=torch.uint8, device="cpu")
        self.__empty = (a, e, m,)
        self.__last = [(a, e, m,)]

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        wait = parse_parameter(Lexicon.WAIT, kw, False, EnumConvertType.BOOLEAN)
        if wait:
            return self.__last
        images = []
        batch_size, rate = parse_parameter(Lexicon.BATCH, kw, (1, 30), EnumConvertType.VEC2INT, 1)[0]
        pbar = ProgressBar(batch_size)
        rate = 1. / rate
        width, height = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT)[0]
        matte = parse_parameter(Lexicon.MATTE, kw, (0,0,0,255), EnumConvertType.VEC4INT, 0, 255)[0]
        mode = parse_parameter(Lexicon.MODE, kw, EnumScaleMode.NONE.name, EnumConvertType.STRING)
        mode = EnumScaleMode[mode]
        sample = parse_parameter(Lexicon.SAMPLE, kw, EnumInterpolation.LANCZOS4.name, EnumConvertType.STRING)
        sample = EnumInterpolation[sample]
        source = parse_parameter(Lexicon.SOURCE, kw, EnumStreamType.URL.name, EnumConvertType.STRING)
        source = EnumStreamType[source]
        if source == EnumStreamType.MONITOR:
            self.__deviceType = EnumStreamType.MONITOR
            which = parse_parameter(Lexicon.MONITOR, kw, "0", EnumConvertType.STRING)
            which = int(which.split('-')[0].strip()) + 1
            for idx in range(batch_size):
                img = monitor_capture(which)
                if img is None:
                    img = channel_solid(width, height, matte)
                else:
                    img = image_scalefit(img, width, height, mode, sample, matte)

                images.append(cv2tensor_full(img))
                if batch_size > 1:
                    pbar.update_absolute(idx)
                    time.sleep(rate)

        elif source == EnumStreamType.WINDOW:
            self.__deviceType = EnumStreamType.WINDOW
            if (which := parse_parameter(Lexicon.WINDOW, kw, "NONE", EnumConvertType.STRING)) != "NONE":
                which = int(which.split('-')[-1].strip())
                dpi = parse_parameter(Lexicon.DPI, kw, True, EnumConvertType.BOOLEAN)
                for idx in range(batch_size):
                    img = window_capture(which, dpi=dpi)
                    if img is None:
                        img = channel_solid(width, height, matte)
                    else:
                        img = image_scalefit(img, width, height, mode, sample, matte)
                    images.append(cv2tensor_full(img))
                    if batch_size > 1:
                        pbar.update_absolute(idx)
                        time.sleep(rate)

        elif source in [EnumStreamType.URL, EnumStreamType.CAMERA]:
            url = parse_parameter(Lexicon.URL, kw, "", EnumConvertType.STRING)
            if source == EnumStreamType.CAMERA:
                url = parse_parameter(Lexicon.CAMERA, kw, "", EnumConvertType.STRING)
                url = url.split('-')[0].strip()
                try:
                    _ = int(url)
                    url = str(url)
                except: url = ""

            if self.__capturing == 0 and (self.__device is None or
                                            self.__deviceType != EnumStreamType.URL or
                                            url != self.__url):
                self.__capturing = time.perf_counter()
                self.__url = url
                try:
                    self.__device = StreamManager().capture(url)
                except Exception as e:
                    logger.error(str(e))

            self.__deviceType = EnumStreamType.URL

            # timeout and try again?
            if self.__capturing > 0 and time.perf_counter() - self.__capturing > 3000:
                logger.error(f'timed out {self.__url}')
                self.__capturing = 0
                self.__url = ""

            if self.__device is not None:
                self.__capturing = 0

                if wait:
                    self.__device.pause()
                else:
                    self.__device.play()

                fps = parse_parameter(Lexicon.FPS, kw, 30, EnumConvertType.INT)
                # if self.__device.fps != fps:
                self.__device.fps = fps

                if type(self.__device) == MediaStreamDevice:
                    self.__device.zoom = parse_parameter(Lexicon.ZOOM, kw, 0, EnumConvertType.INT)

                orient = parse_parameter(Lexicon.ORIENT, kw, EnumCanvasOrientation.NORMAL.name, EnumConvertType.STRING)
                # orient = EnumCanvasOrientation[orient]
                for idx in range(batch_size):
                    img = self.__device.frame
                    if img is None:
                        images.append(self.__empty)
                    else:
                        if type(self.__device) == MediaStreamDevice:
                            if orient in [EnumCanvasOrientation.FLIPX, EnumCanvasOrientation.FLIPXY]:
                                img = cv2.flip(img, 1)
                            if orient in [EnumCanvasOrientation.FLIPY, EnumCanvasOrientation.FLIPXY]:
                                img = cv2.flip(img, 0)
                        img = image_scalefit(img, width, height, mode, sample, matte)
                        images.append(cv2tensor_full(img))
                    pbar.update_absolute(idx)
                    if batch_size > 1:
                        time.sleep(rate)

        elif source == EnumStreamType.SPOUT:
            url = parse_parameter(Lexicon.URL, kw, "", EnumConvertType.STRING)
            if self.__device is None or self.__deviceType != EnumStreamType.SPOUT:
                self.__device = MediaStreamSpout(url)
            self.__deviceType = EnumStreamType.SPOUT
            if self.__device:
                self.__device.url = url
                fps = parse_parameter(Lexicon.FPS, kw, 30, EnumConvertType.INT)
                self.__device.fps = fps
                for idx in range(batch_size):
                    img = self.__device.frame
                    if img is None:
                        images.append(self.__empty)
                    else:
                        img = image_scalefit(img, width, height, mode, sample, matte)
                        images.append(cv2tensor_full(img))
                    pbar.update_absolute(idx)
                    if batch_size > 1:
                        time.sleep(rate)

        if len(images) == 0:
            images.append(self.__empty)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class StreamWriterNode(JOVBaseNode):
    NAME = "STREAM WRITER (JOV) 🎞️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#%EF%B8%8F-stream-writer"
    DESC = "Broadcast ComfyUI Node outputs to custom webserver endpoint."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    INPUT_IS_LIST = False
    OUTPUT_NODE = True
    SORT = 70
    OUT_MAP = {}

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {} ,
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.ROUTE: ("STRING", {"default": "/stream"}),
            Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    """
    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")
    """

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__route = ""
        self.__unique = uuid.uuid4()
        self.__device = None
        self.__starting = False

    def run(self, **kw) -> tuple[torch.Tensor]:
        if self.__starting:
            return
        matte = parse_parameter(Lexicon.MATTE, kw, (0,0,0,255))[0]
        wihi = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT, 1)[0]
        w, h = wihi
        img = parse_parameter(Lexicon.PIXEL, kw, None, EnumConvertType.IMAGE)
        img = tensor2cv(img, width=w, height=h)
        route = parse_parameter(Lexicon.ROUTE, kw, "/stream", EnumConvertType.STRING)
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
        matte = pixel_eval(matte, EnumImageType.BGRA)
        if self.__device is not None:
            mode = parse_parameter(Lexicon.MODE, kw, EnumScaleMode.NONE.name, EnumConvertType.STRING)
            sample = parse_parameter(Lexicon.SAMPLE, kw, EnumInterpolation.LANCZOS4.name, EnumConvertType.STRING)
            #img = image_scalefit(img, w, h, EnumScaleMode.NONE)
            img = image_scalefit(img, w, h, mode, sample, matte)
            self.__device.image = img
        return ()

if JOV_SPOUT:
    class SpoutWriterNode(JOVBaseNode):
        NAME = "SPOUT WRITER (JOV) 🎥"
        CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
        HELP_URL = f"DEVICE#-spout-writer"
        DESC = "Send image data to Spout endpoints"
        DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
        INPUT_IS_LIST = False
        RETURN_TYPES = ("IMAGE", )
        RETURN_NAMES = (Lexicon.IMAGE,)
        OUTPUT_NODE = True
        SORT = 90

        @classmethod
        def INPUT_TYPES(cls) -> dict:
            d = {
            "required": {} ,
            "optional": {
                Lexicon.PIXEL: (WILDCARD, {}),
                Lexicon.ROUTE: ("STRING", {"default": "Spout Sender"}),
                Lexicon.FPS: ("INT", {"min": 1, "max": 60, "default": 30}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
                Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
            }}
            return Lexicon._parse(d, cls.HELP_URL)

        @classmethod
        def IS_CHANGED(cls, **kw) -> float:
            return float("nan")

        def __init__(self, *arg, **kw) -> None:
            super().__init__(*arg, **kw)
            self.__sender = SpoutSender("")

        def run(self, **kw) -> tuple[torch.Tensor]:
            pA = parse_parameter(Lexicon.PIXEL, kw, None, EnumConvertType.IMAGE)
            host = parse_parameter(Lexicon.ROUTE, kw, "", EnumConvertType.STRING)[0]
            fps = parse_parameter(Lexicon.FPS, kw, 30, EnumConvertType.INT)[0]
            delta = 1. / float(fps)
            mode = parse_parameter(Lexicon.MODE, kw, EnumScaleMode.NONE.name, EnumConvertType.STRING)[0]
            wihi = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT, 1)[0]
            sample = parse_parameter(Lexicon.SAMPLE, kw, EnumInterpolation.LANCZOS4.name, EnumConvertType.STRING)[0]
            matte = parse_parameter(Lexicon.MATTE, kw, (0,0,0,255), EnumConvertType.VEC4INT, 0, 255)[0]
            images = []
            #params = [tuple(x) for x in zip_longest_fill(pA, host, delta, mode, wihi, sample, matte)]
            pbar = ProgressBar(len(pA))
            for idx, pA in enumerate(pA):
                self.__sender.host = host
                width, height = wihi
                matte = pixel_eval(matte, EnumImageType.BGRA)
                image = tensor2cv(pA)
                image = image_scalefit(image, width, height, mode, sample, matte)
                image = image_convert(image, 4)
                images.append(cv2tensor(image))
                if pA is not None:
                    image[:, :, [0, 2]] = image[:, :, [2, 0]]
                    self.__sender.frame = image
                time.sleep(delta)
                pbar.update_absolute(idx)
            return images

class MIDIMessageNode(JOVBaseNode):
    NAME = "MIDI MESSAGE (JOV) 🎛️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#%EF%B8%8F-midi-message"
    DESC = "Expands a MIDI message into its values."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.VALUE, Lexicon.NORMALIZE, )
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {} ,
            "optional": {
            Lexicon.MIDI: ('JMIDIMSG', {"default": None})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[object, bool, int, int, int, float, float]:
        message = parse_parameter(Lexicon.MIDI, kw, None, EnumConvertType.ANY)
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
    HELP_URL = f"{JOV_CATEGORY}#-midi-reader"
    DESC = "Capture MIDI devices and pass the data into Comfy."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT', 'FLOAT',)
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.VALUE, Lexicon.NORMALIZE,)
    SORT = 5
    DEVICES = midi_device_names()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {} ,
            "optional": {
            Lexicon.DEVICE : (cls.DEVICES, {"default": cls.DEVICES[0] if len(cls.DEVICES) > 0 else None})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

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
            case "note_off":
                self.__note = data.note
                self.__value = data.velocity

    def run(self, **kw) -> tuple[bool, int, int, int]:
        device = parse_parameter(Lexicon.DEVICE, kw, None, EnumConvertType.STRING)
        if device != self.__device:
            self.__q_in.put(device)
            self.__device = device
        normalize = self.__value / 127.
        msg = MIDIMessage(self.__note_on, self.__channel, self.__control, self.__note, self.__value)
        return (msg, self.__note_on, self.__channel, self.__control, self.__note, self.__value, normalize,  )

class MIDIFilterEZNode(JOVBaseNode):
    NAME = "MIDI FILTER EZ (JOV) ❇️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#%EF%B8%8F-midi-filter-ez"
    DESC = "Filter MIDI messages by channel, message type or value."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    INPUT_IS_LIST = False
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN',)
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.TRIGGER,)
    SORT = 25

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
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[bool]:
        message = parse_parameter(Lexicon.MIDI, kw, None, EnumConvertType.ANY)[0]
        if message is None:
            logger.warning('no midi message. connected?')
            return (message, False, )

        # empty values mean pass-thru (no filter)
        val = parse_parameter(Lexicon.MODE, kw, MIDINoteOnFilter.IGNORE.name, EnumConvertType.STRING)
        val = MIDINoteOnFilter[val]
        if val != MIDINoteOnFilter.IGNORE:
            if val == MIDINoteOnFilter.NOTE_ON and message.note_on != True:
                return (message, False, )
            if val == MIDINoteOnFilter.NOTE_OFF and message.note_on != False:
                return (message, False, )

        if (val := parse_parameter(Lexicon.CHANNEL, kw, -1, EnumConvertType.INT)[0]) != -1 and val != message.channel:
            return (message, False, )
        if (val := parse_parameter(Lexicon.CONTROL, kw, -1, EnumConvertType.INT)[0]) != -1 and val != message.control:
            return (message, False, )
        if (val := parse_parameter(Lexicon.NOTE, kw, -1, EnumConvertType.INT)[0]) != -1 and val != message.note:
            return (message, False, )
        if (val := parse_parameter(Lexicon.VALUE, kw, -1, EnumConvertType.INT)[0]) != -1 and val != message.value:
            return (message, False, )
        if (val := parse_parameter(Lexicon.NORMALIZE, kw, -1, EnumConvertType.INT)[0]) != -1 and isclose(val, message.normal):
            return (message, False, )
        return (message, True, )

class MIDIFilterNode(JOVBaseNode):
    NAME = "MIDI FILTER (JOV) ✳️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#%EF%B8%8F-midi-filter"
    DESC = "Filter MIDI messages by channel, message type or value."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    INPUT_IS_LIST = False
    RETURN_TYPES = ('JMIDIMSG', 'BOOLEAN', )
    RETURN_NAMES = (Lexicon.MIDI, Lexicon.TRIGGER,)
    SORT = 20
    EPSILON = 1e-6

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
        }}
        return Lexicon._parse(d, cls.HELP_URL)

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
        message = parse_parameter(Lexicon.MIDI, kw, None, EnumConvertType.ANY)[0]
        if message is None:
            logger.warning('no midi message. connected?')
            return (message, False, )

        # empty values mean pass-thru (no filter)
        val = parse_parameter(Lexicon.ON, kw, MIDINoteOnFilter.IGNORE.name, EnumConvertType.STRING)[0]
        val = MIDINoteOnFilter[val]
        if val != MIDINoteOnFilter.IGNORE:
            if val == "TRUE" and message.note_on != True:
                return (message, False, )
            if val == "FALSE" and message.note_on != False:
                return (message, False, )
        if self.__filter(parse_parameter(Lexicon.CHANNEL, kw, False, EnumConvertType.BOOLEAN)[0], message.channel) == False:
            return (message, False, )
        if self.__filter(parse_parameter(Lexicon.CONTROL, kw, False, EnumConvertType.BOOLEAN)[0], message.control) == False:
            return (message, False, )
        if self.__filter(parse_parameter(Lexicon.NOTE, kw, False, EnumConvertType.BOOLEAN)[0], message.note) == False:
            return (message, False, )
        if self.__filter(parse_parameter(Lexicon.VALUE, kw, False, EnumConvertType.BOOLEAN)[0], message.value) == False:
            return (message, False, )
        if self.__filter(parse_parameter(Lexicon.NORMALIZE, kw, False, EnumConvertType.BOOLEAN)[0], message.normal) == False:
            return (message, False, )
        return (message, True, )

class AudioDeviceNode(JOVBaseNode):
    NAME = "AUDIO DEVICE (JOV) 📺"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    HELP_URL = f"{JOV_CATEGORY}#-audio-device"
    DESC = "Stream from System audio devices into ComfyUI workflows"
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    INPUT_IS_LIST = False
    RETURN_TYPES = ('WAVE',)
    RETURN_NAMES = (Lexicon.WAVE,)
    SORT = 90

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        dev = AudioDevice()
        dev_list = list(dev.devices.keys())
        d = {
            "required": {} ,
            "optional": {
            Lexicon.DEVICE: (dev_list, {"default": next(iter(dev_list))}),
            Lexicon.TRIGGER: ("BOOLEAN", {"default": True, "tooltip":"Auto-record when executed by the Q"}),
            Lexicon.RECORD: ("BOOLEAN", {"default": True, "tooltip":"Control to manually adjust when the selected device is recording"}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        wave = None
        return wave
