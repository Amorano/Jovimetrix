"""
Jovimetrix - Device -- WEBCAM, REMOTE URLS, SPOUT
"""

import sys
import time
import uuid
from typing import Tuple
from enum import Enum

import cv2
import torch
from loguru import logger

from comfy.utils import ProgressBar

from .. import \
    JOV_DOCKERENV, JOV_TYPE_IMAGE, \
    InputType, JOVBaseNode, JOVImageNode, Lexicon, RGBAMaskType, \
    deep_merge

from ..sup.util import \
    EnumConvertType, \
    parse_param, zip_longest_fill

from ..sup.stream import \
    JOV_SPOUT, \
    StreamingServer, StreamManager, MediaStreamDevice, \
    camera_list, monitor_list, window_list, monitor_capture

if not JOV_DOCKERENV:
    from ..sup.stream import window_capture

from ..sup.image.adjust import \
    EnumScaleMode, EnumInterpolation, \
    image_scalefit

from ..sup.image.channel import channel_solid

if JOV_SPOUT:
    from ..sup.stream import SpoutSender, MediaStreamSpout

from ..sup.image import \
    MIN_IMAGE_SIZE, \
    EnumImageType, \
    image_convert, cv2tensor_full, tensor2cv

from ..sup.image.color import pixel_eval

# ==============================================================================

JOV_CATEGORY = "DEVICE"

# ==============================================================================
# == ENUMERATION ==
# ==============================================================================

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

# ==============================================================================

class StreamReaderNode(JOVImageNode):
    NAME = "STREAM READER (JOV) 📺"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 50
    CAMERAS = None
    DESCRIPTION = """
Capture frames from various sources such as URLs, cameras, monitors, windows, or Spout streams. It supports batch processing, allowing multiple frames to be captured simultaneously. The node provides options for configuring the source, resolution, frame rate, zoom, orientation, and interpolation method. Additionally, it supports capturing frames from multiple monitors or windows simultaneously. The captured frames are returned as tensors, enabling further processing downstream.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()

        if cls.CAMERAS is None:
            cls.CAMERAS = [f"{i} - {v['w']}x{v['h']}" for i, v in enumerate(camera_list().values())]
        camera_default = cls.CAMERAS[0] if len(cls.CAMERAS) else "NONE"

        monitor = ["NONE"]
        try:
            monitors = monitor_list()
            if monitors:  # Check if the list is not empty
                monitors.pop(0)
            monitor = [f"{i} - {v['width']}x{v['height']}" for i, v in enumerate(monitors.values())]
        except:
            pass

        # would be empty if monitors.values() is empty
        if len(monitor) == 0:
            monitor = ["NONE"]

        window = []
        if sys.platform.startswith('win'):
            window = [f"{v} - {k}" for k, v in window_list().items()]
        window_default = window[0] if len(window) else "NONE"

        names = EnumStreamType._member_names_
        if not JOV_SPOUT:
            if names: # Check if the list is not empty
                names.pop()

        d = deep_merge(d, {
            "optional": {
                Lexicon.SOURCE: (names, {"default": EnumStreamType.URL.name}),
                Lexicon.URL: ("STRING", {"default": "", "dynamicPrompts": False}),
                Lexicon.CAMERA: (cls.CAMERAS, {"default": camera_default, "choice": "list of system streaming devices"}),
                Lexicon.MONITOR: (monitor, {"default": monitor[0], "choice": "list of system monitor devices"}),
                Lexicon.WINDOW: (window, {"default": window_default, "choice": "list of available system windows"}),
                Lexicon.DPI: ("BOOLEAN", {"default": True}),
                Lexicon.BBOX: ("VEC4", {"default": (0, 0, 1, 1), "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
                Lexicon.FPS: ("INT", {"min": 1, "max": 60, "default": 30}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
                Lexicon.BATCH: ("VEC2INT", {"default": (1, 30), "label": ["COUNT", "FPS"], "tooltip": "Number of frames wanted and the FPS"}),
                Lexicon.ORIENT: (EnumCanvasOrientation._member_names_, {"default": EnumCanvasOrientation.NORMAL.name}),
                Lexicon.ZOOM: ("FLOAT", {"min": 0, "max": 1, "step": 0.005, "default": 0.}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })

        if JOV_DOCKERENV:
            d["optional"].pop(Lexicon.MONITOR)
            d["optional"].pop(Lexicon.WINDOW)

        return Lexicon._parse(d)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("NaN")

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

    def run(self, *arg, **kw) -> RGBAMaskType:
        wait = parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False)[0]
        if wait:
            return self.__last
        images = []
        batch_size, rate = parse_param(kw, Lexicon.BATCH, EnumConvertType.VEC2INT, [(1, 30)], 1)[0]
        pbar = ProgressBar(batch_size)
        rate = 1. / rate
        width, height = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)])[0]
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0,0,0,255)], 0, 255)[0]
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)[0]
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)[0]
        source = parse_param(kw, Lexicon.SOURCE, EnumStreamType, EnumStreamType.URL.name)[0]
        if source == EnumStreamType.MONITOR:
            self.__deviceType = EnumStreamType.MONITOR
            if (which := parse_param(kw, Lexicon.MONITOR, EnumConvertType.STRING, "NONE")[0]) != "NONE":
                which = int(which.split('-')[0].strip()) + 1
                bbox = parse_param(kw, Lexicon.BBOX, EnumConvertType.VEC4, [(0,0,1,1)], 0, 1)[0]
                for idx in range(batch_size):

                    img = monitor_capture(which, bbox)
                    if img is None:
                        img = channel_solid(width, height, matte)
                    elif mode != EnumScaleMode.MATTE:
                        img = image_scalefit(img, width, height, mode, sample, matte)

                    images.append(cv2tensor_full(img))
                    if batch_size > 1:
                        pbar.update_absolute(idx)
                        time.sleep(rate)

        elif source == EnumStreamType.WINDOW:
            self.__deviceType = EnumStreamType.WINDOW
            if (which := parse_param(kw, Lexicon.WINDOW, EnumConvertType.STRING, "NONE")[0]) != "NONE":
                which = int(which.split('-')[-1].strip())
                dpi = parse_param(kw, Lexicon.DPI, EnumConvertType.BOOLEAN, True)[0]
                for idx in range(batch_size):
                    img = window_capture(which, dpi=dpi)
                    if img is None:
                        img = channel_solid(width, height, matte)
                    elif mode != EnumScaleMode.MATTE:
                        img = image_scalefit(img, width, height, mode, sample, matte)
                    images.append(cv2tensor_full(img))
                    if batch_size > 1:
                        pbar.update_absolute(idx)
                        time.sleep(rate)

        elif source in [EnumStreamType.URL, EnumStreamType.CAMERA]:
            url = parse_param(kw, Lexicon.URL, EnumConvertType.STRING, "")[0]
            if source == EnumStreamType.CAMERA:
                url = parse_param(kw, Lexicon.CAMERA, EnumConvertType.STRING, "")[0]
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

                fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 30)[0]
                # if self.__device.fps != fps:
                self.__device.fps = fps

                if type(self.__device) == MediaStreamDevice:
                    self.__device.zoom = parse_param(kw, Lexicon.ZOOM, EnumConvertType.FLOAT, 0, 0, 1)[0]

                orient = parse_param(kw, Lexicon.ORIENT, EnumCanvasOrientation, EnumCanvasOrientation.NORMAL.name)[0]

                self.__device

                for idx in range(batch_size):
                    img = self.__device.frame
                    if img is None:
                        images.append(self.__empty)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
                        if type(self.__device) == MediaStreamDevice:
                            if orient in [EnumCanvasOrientation.FLIPX, EnumCanvasOrientation.FLIPXY]:
                                img = cv2.flip(img, 1)
                            if orient in [EnumCanvasOrientation.FLIPY, EnumCanvasOrientation.FLIPXY]:
                                img = cv2.flip(img, 0)
                        if mode != EnumScaleMode.MATTE:
                            img = image_scalefit(img, width, height, mode, sample, matte)
                        images.append(cv2tensor_full(img))
                    pbar.update_absolute(idx)
                    if batch_size > 1:
                        time.sleep(rate)

        elif source == EnumStreamType.SPOUT:
            url = parse_param(kw, Lexicon.URL, EnumConvertType.STRING, "")[0]
            if self.__device is None or self.__deviceType != EnumStreamType.SPOUT:
                self.__device = MediaStreamSpout(url)
            self.__deviceType = EnumStreamType.SPOUT
            if self.__device:
                self.__device.url = url
                fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 30)[0]
                self.__device.fps = fps
                for idx in range(batch_size):
                    img = self.__device.frame
                    if img is None:
                        images.append(self.__empty)
                    elif mode != EnumScaleMode.MATTE:
                        img = image_scalefit(img, width, height, mode, sample, matte)
                        images.append(cv2tensor_full(img))
                    pbar.update_absolute(idx)
                    if batch_size > 1:
                        time.sleep(rate)

        if len(images) == 0:
            images.append(self.__empty)
        self.__last = [torch.stack(i) for i in zip(*images)]
        return self.__last

class StreamWriterNode(JOVBaseNode):
    NAME = "STREAM WRITER (JOV) 🎞️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    OUTPUT_NODE = True
    SORT = 70
    OUT_MAP = {}
    DESCRIPTION = """
Sends frames to a specified route, typically for live streaming or recording purposes. It accepts tensors representing images and allows configuration of parameters such as route, resolution, scaling mode, interpolation method, and matte color. The node continuously streams frames to the specified route, enabling real-time visualization or recording of processed video data.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.ROUTE: ("STRING", {"default": "/stream"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 0), "rgb": True})
            }
        })
        return Lexicon._parse(d)

    """
    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("NaN")
    """

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__route = ""
        self.__unique = uuid.uuid4()

        self.__device = StreamManager().capture(self.__unique, static=True)

    def run(self, **kw) -> Tuple[torch.Tensor]:
        route = parse_param(kw, Lexicon.ROUTE, EnumConvertType.STRING, "/stream")
        images = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0,0,0,0)], 0, 255)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        params = list(zip_longest_fill(route, images, wihi, matte, mode, sample))
        pbar = ProgressBar(len(params))
        for idx, (route, images, wihi, matte, mode, sample) in enumerate(params):
            if route != self.__route:
                try:
                    StreamingServer().endpointAdd(route, self.__device)
                except Exception as e:
                    logger.error(e)
                StreamWriterNode.OUT_MAP[route] = self.__device
                self.__route = route

            if self.__device is not None:
                w, h = wihi
                matte = pixel_eval(matte, EnumImageType.BGRA)
                if images is None:
                    img = channel_solid(w, h, matte, EnumImageType.RGBA)
                else:
                    img = tensor2cv(images)
                    if mode != EnumScaleMode.MATTE:
                        img = image_scalefit(img, w, h, mode, sample, matte)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.__device.image = img
            pbar.update_absolute(idx)
        return ()

if JOV_SPOUT:
    class SpoutWriterNode(JOVBaseNode):
        NAME = "SPOUT WRITER (JOV) 🎥"
        CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
        RETURN_TYPES = ()
        OUTPUT_NODE = True
        SORT = 90
        DESCRIPTION = """
Sends frames to a specified Spout receiver application for real-time video sharing. Accepts tensors representing images and allows configuration of parameters such as the Spout host, frame rate, resolution, scaling mode, interpolation method, and matte color. The node continuously streams frames to the specified Spout host, enabling real-time visualization or integration with other applications that support Spout.
"""

        @classmethod
        def INPUT_TYPES(cls) -> InputType:
            d = super().INPUT_TYPES()
            d = deep_merge(d, {
                "optional": {
                    Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                    Lexicon.ROUTE: ("STRING", {"default": "Spout Sender"}),
                    Lexicon.FPS: ("INT", {"min": 0, "max": 60, "default": 30, "tooltip": "@@@ NOT USED @@@"}),
                    Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                    Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                    Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                    Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
                }
            })
            return Lexicon._parse(d)

        @classmethod
        def IS_CHANGED(cls, **kw) -> float:
            return float("NaN")

        def __init__(self, *arg, **kw) -> None:
            super().__init__(*arg, **kw)
            self.__sender = SpoutSender("")

        def run(self, **kw) -> Tuple[torch.Tensor]:
            images = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
            host = parse_param(kw, Lexicon.ROUTE, EnumConvertType.STRING, "")
            #fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 30)
            mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
            wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
            sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
            matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0,0,0,0)], 0, 255)
            # results = []
            params = list(zip_longest_fill(images, host, mode, wihi, sample, matte))
            pbar = ProgressBar(len(params))
            for idx, (img, host, mode, wihi, sample, matte) in enumerate(params):
                self.__sender.host = host
                matte = pixel_eval(matte, EnumImageType.BGRA)
                w, h = wihi
                img = channel_solid(w, h, chan=EnumImageType.BGRA) if img is None else tensor2cv(img)
                if mode != EnumScaleMode.MATTE:
                    img = image_scalefit(img, w, h, mode, sample, matte)
                img = image_convert(img, 4)
                self.__sender.frame = img
                pbar.update_absolute(idx)
            return ()
