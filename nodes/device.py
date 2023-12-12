"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Device -- MIDI, WEBCAM

    type 0 (single track): all messages are saved in one track
    type 1 (synchronous): all tracks start at the same time
    type 2 (asynchronous): each track is independent of the others
"""

import uuid

import cv2
import torch
import numpy as np

from Jovimetrix import deep_merge_dict, tensor2cv, cv2mask, cv2tensor, \
        JOVBaseNode, JOVImageBaseNode, Logger, \
        IT_PIXELS, IT_ORIENT, IT_CAM, IT_REQUIRED, \
        IT_WHMODE, IT_REQUIRED, MIN_HEIGHT, MIN_WIDTH, IT_INVERT

from Jovimetrix.sup.comp import image_grid, light_invert, geo_scalefit, EnumInterpolation, IT_SAMPLE
from Jovimetrix.sup.stream import StreamingServer, StreamManager

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
    print(mid)
    for msg in mid.tracks[0]:
        print(msg)

def print_message(message) -> None:
    print(message)

def poll_midi() -> None:
    while 1:
        with mido.open_input(callback=print_message) as inport:
            for msg in inport:
                print(msg)

# =============================================================================

class StreamReaderNode(JOVImageBaseNode):
    NAME = "📺 StreamReader (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False, )

    EMPTY = np.zeros((MIN_HEIGHT, MIN_WIDTH, 3), dtype=np.float32)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "url": ("STRING", {"default": 0}),
            },
            "optional": {
                "fps": ("INT", {"min": 1, "max": 60, "step": 1, "default": 60}),
                "hold": ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(d, IT_WHMODE, IT_SAMPLE, IT_INVERT, IT_ORIENT, IT_CAM)

    @classmethod
    def IS_CHANGED(cls, url: str, width: int, height: int, fps: float,
                   hold: bool, sample: str, zoom: float, **kw) -> float:

        if (device := StreamManager.capture(url)) is None:
            raise Exception(f"stream failed {url}")

        if device.width != width or device.height != height:
            device.sizer(width, height, sample)

        if device.zoom != zoom:
            device.zoom = zoom

        if hold:
            device.pause()
        else:
            device.play()

        if device.fps != fps:
            device.fps = fps

        return float("nan")

    def __init__(self) -> None:
        self.__device = None
        self.__url = ""
        self.__last = StreamReaderNode.EMPTY

    def run(self, url: str, fps: float, hold: bool, width: int,
            height: int, mode: str, resample: str, invert: float, orient: str,
            zoom: float) -> tuple[torch.Tensor, torch.Tensor]:

        if self.__device is None or self.__device.captured or url != self.__url:
            self.__device = StreamManager.capture(url)
            if self.__device is None or not self.__device.captured:
                return (cv2tensor(self.__last),
                        cv2mask(self.__last),
                )

        ret, image = self.__device.frame
        self.__last = image = image if image is not None else self.__last
        if ret:
            h, w = self.__last.shape[:2]
            if width != w or height != h:
                rs = EnumInterpolation[resample].value
                self.__device.sizer(width, height, rs)

            if orient in ["FLIPX", "FLIPXY"]:
                image = cv2.flip(image, 1)

            if orient in ["FLIPY", "FLIPXY"]:
                image = cv2.flip(image, 0)

            if invert != 0.:
                image = light_invert(image, invert)

        return (
            cv2tensor(image),
            cv2mask(image)
        )

class StreamWriterNode(JOVBaseNode):
    OUT_MAP = {}

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "route": ("STRING", {"default": "/stream"}),
            },
            "optional": {
                "hold": ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODE, IT_INVERT)

    NAME = "🎞️ StreamWriter (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (None, None,)

    @classmethod
    def IS_CHANGED(cls, route: str, hold: bool, width: int, height: int, fps: float, **kw) -> float:

        if (device := StreamManager.capture(route, static=True)) is None:
            raise Exception(f"stream failed {route}")

        if device.size[0] != width or device.size[1] != height:
            device.size = (width, height)

        if hold:
            device.pause()
        else:
            device.play()

        if device.fps != fps:
            device.fps = fps
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super(StreamWriterNode).__init__(self, *arg, **kw)
        self.__ss = StreamingServer()
        self.__route = ""
        self.__unique = uuid.uuid4()
        self.__device = None
        StreamWriterNode.OUT_MAP[self.__unique] = None

    def run(self, pixels: list[torch.Tensor], route: list[str],
            hold: list[bool], width: list[int], height: list[int],
            mode: list[str],
            resample: list[str],
            invert: list[float]) -> torch.Tensor:

        route = route[0]
        hold = hold[0]
        Logger.debug(self.NAME, route)

        if route != self.__route:
            # close old, if any
            if self.__device:
                self.__device.release()

            # startup server
            self.__device = StreamManager.capture(self.__unique, static=True)
            self.__ss.endpointAdd(route, self.__device)
            self.__route = route
            Logger.debug(self.NAME, "START", route)

        w = width[min(idx, len(width)-1)]
        h = height[min(idx, len(height)-1)]
        m = mode[0]
        rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
        out = []

        stride = len(pixels)
        grid = int(np.sqrt(stride))
        if grid * grid < stride:
            grid += 1
        sw, sh = w // stride, h // stride

        for idx, image in enumerate(pixels):
            image = tensor2cv(image)
            image = geo_scalefit(image, sw, sh, m, rs)
            i = invert[min(idx, len(invert)-1)]
            if i != 0:
                image = light_invert(image, i)
            out.append(image)

        image = image_grid(out, w, h)
        image = geo_scalefit(image, w, h, m, rs)
        self.__device.post(image)

class MIDIPortNode(JOVBaseNode):
    NAME = "🎹 MIDI Port (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = "Reads input from a midi device"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,)
    RETURN_TYPES = ('FLOAT',)
    RETURN_NAMES = ("🎛️",)

    @classmethod
    def INPUT_TYPES(s) -> dict:
        d = {"optional": {
            "channel" : ("INT", {"default":0}),
            "port" : ("INT", {"default":0}),
        }}
        return deep_merge_dict(IT_REQUIRED, d)

    def run(self, channel:int=0,  port:int=0) -> tuple[float]:
        val = 0.
        return (val, )

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
   m = mido.get_output_names()
   print(m)

   poll_midi()