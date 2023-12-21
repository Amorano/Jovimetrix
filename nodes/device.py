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
from queue import Queue, Empty

import cv2
import torch
import numpy as np

from Jovimetrix import deep_merge_dict, tensor2cv, cv2mask, cv2tensor, zip_longest_fill, \
        JOVBaseNode, JOVImageBaseNode, JOVImageInOutBaseNode, Logger, Lexicon, EnumCanvasOrientation, \
        IT_PIXELS, IT_ORIENT, IT_CAM, IT_WHMODE, IT_REQUIRED, IT_INVERT

from Jovimetrix.sup.comp import image_grid, light_invert, geo_scalefit, \
    EnumInterpolation, \
    IT_SAMPLE
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
                Logger.err(str(e))

# =============================================================================

class StreamReaderNode(JOVImageBaseNode):
    NAME = "STREAM READER (JOV) 📺"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )
    EMPTY = np.zeros((64, 64, 3), dtype=np.float32)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.URL: ("STRING", {"default": 0}),
                Lexicon.FPS: ("INT", {"min": 1, "max": 60, "step": 1, "default": 60}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WHMODE, IT_SAMPLE, IT_INVERT, IT_ORIENT, IT_CAM)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        url = kw.get(Lexicon.URL, False)
        if (device := StreamManager.capture(url)) is None:
            raise Exception(f"stream failed {url}")
        fps = kw.get(Lexicon.FPS, 60)
        wait = kw.get(Lexicon.WAIT, False)
        wh = kw.get(Lexicon.WH, 0)
        zoom = kw.get(Lexicon.ZOOM, 1)
        sample = kw.get(Lexicon.SAMPLE, EnumInterpolation.LANCZOS4)

        width, height = wh
        if device.width != width or device.height != height:
            device.sizer(width, height, sample)

        if device.zoom != zoom:
            device.zoom = zoom

        if wait:
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

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:

        wh = kw.get(Lexicon.WH, 0)
        orient = kw.get(Lexicon.ORIENT, EnumCanvasOrientation)
        invert = kw.get(Lexicon.INVERT, 0)
        url = kw.get(Lexicon.URL, "")
        # fps = kw.get(Lexicon.FPS, 60)
        # mode = kw.get(Lexicon.FPS, EnumInterpolation.LANCZOS4)
        rs = kw.get(Lexicon.FPS, EnumInterpolation.LANCZOS4)

        if self.__device is None or self.__device.captured or url != self.__url:
            self.__device = StreamManager.capture(url)
            if self.__device is None or not self.__device.captured:
                return (cv2tensor(self.__last),
                        cv2mask(self.__last),
                )

        ret, img = self.__device.frame
        self.__last = img = img if img else self.__last
        if ret:
            h, w = self.__last.shape[:2]
            width, height = wh
            if width != w or height != h:
                rs = EnumInterpolation[rs]
                self.__device.sizer(width, height, rs)

            if orient in ["FLIPX", "FLIPXY"]:
                img = cv2.flip(img, 1)

            if orient in ["FLIPY", "FLIPXY"]:
                img = cv2.flip(img, 0)

            if (invert or 0) != 0.:
                img = light_invert(img, invert)

        return (
            cv2tensor(img),
            cv2mask(img)
        )

class StreamWriterNode(JOVImageInOutBaseNode):
    NAME = "STREAM WRITER (JOV) 🎞️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = ""
    OUT_MAP = {}

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.ROUTE: ("STRING", {"default": "/stream"}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_WHMODE, IT_INVERT)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        route = kw.get(Lexicon.ROUTE, [None])
        wh = kw.get(Lexicon.WH, [None])
        wait = kw.get(Lexicon.WAIT, [None])
        fps = kw.get(Lexicon.FPS, [None])
        sample = kw.get(Lexicon.SAMPLE, [None])
        width, height = wh

        if (device := StreamManager.capture(route, static=True)) is None:
            raise Exception(f"stream failed {route}")

        if device.size[0] != width or device.size[1] != height:
            device.size = (width, height)

        if wait:
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

    def run(self, **kw) -> tuple[torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        route = kw.get(Lexicon.ROUTE, [None])
        wait = kw.get(Lexicon.ROUTE, [None])
        mode = kw.get(Lexicon.MODE, [None])
        wihi = kw.get(Lexicon.WH, [None])
        invert = kw.get(Lexicon.INVERT, [None])
        sample = sample or [None]

        for data in zip_longest_fill(pixels, route, wait, wihi, sample, invert):
            img, r, wait, wh, rs, i = data
            w, h = wh
            h = h or 0
            w = w or 0
            img = img if img else np.zeros((h, w, 3), dtype=np.uint8)
            if r != self.__route:
                # close old, if any
                if self.__device:
                    self.__device.release()

                # startup server
                self.__device = StreamManager.capture(self.__unique, static=True)
                self.__ss.endpointAdd(r, self.__device)
                self.__route = r
                Logger.debug(self.NAME, "START", r)

            rs = EnumInterpolation[rs] if rs else EnumInterpolation.LANCZOS4
            out = []

            stride = len(img)
            grid = int(np.sqrt(stride))
            if grid * grid < stride:
                grid += 1
            sw, sh = w // stride, h // stride

            img = tensor2cv(img)
            img = geo_scalefit(img, sw, sh, mode, rs)
            if (i or 0) != 0:
                img = light_invert(img, i)
            out.append(img)

        image = image_grid(out, w, h)
        image = geo_scalefit(image, w, h, mode, rs)
        self.__device.post(image)

class MIDIReaderNode(JOVBaseNode):
    NAME = "MIDI READER (JOV) 🎹"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/DEVICE"
    DESCRIPTION = "Reads input from a midi device"
    OUTPUT_IS_LIST = (False, False, False, False, False)
    RETURN_TYPES = ('BOOLEAN', 'INT', 'INT', 'INT', 'FLOAT')
    RETURN_NAMES = (Lexicon.ON, Lexicon.CHANNEL, Lexicon.CONTROL, Lexicon.NOTE, Lexicon.AMT,)

    DEVICES = mido.get_input_names()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.DEVICE : (cls.DEVICES, {"default": cls.DEVICES[0] if len(cls.DEVICES) > 0 else None}),
                Lexicon.NORMALIZE : ("BOOLEAN", {"default": True}),
                Lexicon.FILTER : ("BOOLEAN", {"default": False}),
                Lexicon.CHANNEL : ("INT", {"default": 0}),
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

        channel = kw.get(Lexicon.CHANNEL, [None])
        normalize = kw.get(Lexicon.NORMALIZE, [None])
        device = kw.get(Lexicon.DEVICE, [None])
        filter = kw.get(Lexicon.FILTER, [None])

        if device != self.__device:
            self.__q_in.put(device)
            self.__device = device

        if filter and self.__channel != channel:
            return (self.__channel, channel, False, 0, 0, 0)

        if (value := self.__value) > 0 and normalize:
            value /= 127.
        Logger.spam(channel, self.__note_on, self.__channel, self.__control, self.__note, value)
        return (self.__note_on, self.__channel, self.__control, self.__note, value)

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


