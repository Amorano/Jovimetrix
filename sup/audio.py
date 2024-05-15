"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Audio Support
"""

import io
import time
import threading
from enum import Enum
from queue import Queue
from urllib.request import urlopen
from typing import OrderedDict, Tuple

import ffmpeg
import pyaudio
import librosa
import numpy as np
from PIL import Image, ImageDraw

from loguru import logger

from Jovimetrix.sup.image import EnumScaleMode, image_scalefit, pil2cv, TYPE_PIXEL

# =============================================================================

class EnumGraphType(Enum):
    NORMAL = 0
    SOUNDCLOUD = 1

# =============================================================================
# === LOADERS ===
# =============================================================================

# FFMPEG... dont love
def load_audio(url) -> np.ndarray[np.int16]:
    cmd = (
        ffmpeg.input(url)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1)
        .run(input=None, capture_stdout=False, capture_stderr=False)
    )
    # logger.debug(url)
    return np.frombuffer(cmd[0], dtype=np.int16)

def load_audio(url: str, sample_rate: int=22050, offset: float=0, mono:bool=True,
               duration: float=None) -> Tuple[np.ndarray[np.int16], float]:

    if duration == 0.0:
        duration = None

    if url.startswith("http"):
        url = io.BytesIO(urlopen(url).read())

    audio, rate = librosa.load(url, sr=sample_rate, offset=offset, duration=duration)
    # audio = torch.from_numpy(audio)[None, :, None]
    return audio, rate

# =============================================================================
# === VISUALIZE ===
# =============================================================================

def graph_sausage(data: np.ndarray, bar_count:int, width:int, height:int,
                    thickness: float = 0.5, offset: float = 0.0,
                    color_line:TYPE_PIXEL=(172, 172, 172, 255),
                    color_back:TYPE_PIXEL=(0, 0, 0, 255)) -> np.ndarray[np.int8]:

    normalized_data = data.astype(np.float32) / 32767.0
    length = len(normalized_data)
    ratio = length / bar_count
    max_array = np.maximum.reduceat(np.abs(normalized_data), np.arange(0, length, ratio, dtype=int))
    highest_line = max_array.max()
    line_width = (width + bar_count) // bar_count
    line_ratio = highest_line / height
    image = Image.new('RGBA', (bar_count * line_width, height), color_back)
    draw = ImageDraw.Draw(image)
    for i, item in enumerate(max_array):
        item_height = item / line_ratio
        current_x = int((i + offset) * line_width)
        current_y = int((height - item_height) / 2)
        draw.line((current_x, current_y, current_x, current_y + item_height),
                  fill=color_line, width=int(thickness * line_width))
    image = pil2cv(image)
    return image_scalefit(image, width, height, EnumScaleMode.FIT)

# =============================================================================
# === DEVICES ===
# =============================================================================

class AudioDevice:
    def __init__(self) -> None:
        self.__recording = False
        self.__thread_running = False
        self.__thread = None
        self.__p = pyaudio.PyAudio()
        count = self.__p.get_device_count()
        self.__devices = OrderedDict()
        for i in range(count):
            info = self.__p.get_device_info_by_index(i)
            self.__devices[info['name']] = info
        self.__device = None
        self.__rate = 44100
        self.__chunk = 1024
        self.__format = pyaudio.paInt16
        self.__q = Queue()
        self.__thread = threading.Thread(target=self.__record, daemon=True)
        self.__thread.start()
        self.__thread_running = True

    @property
    def buffer(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The current recorded audio buffer.
        Returns the full mix with left and right channels.
        """
        count = self.__q.qsize()
        frames = [self.__q.get() for _ in range(count)]
        if len(frames) == 0:
            return None
        frames = b''.join(frames)
        mix = np.frombuffer(frames, dtype=np.int16)
        return mix, mix[::2], mix[1::2]

    @property
    def devices(self) -> dict:
        return self.__devices

    @property
    def device(self) -> str:
        return self.__device

    @device.setter
    def device(self, device:str) -> None:
        if (device := self.__devices.get(device, None)) is None:
            logger.warning(f"Device doesn't exist {device}")
            return
        self.__device = device

    def record(self) -> None:
        self.__recording = True

    def __record(self) -> None:
        while self.__thread_running:
            if not self.__recording:
                time.sleep(0.05)
                continue

            channels, as_loopback = 2, False
            if self.__device['maxInputChannels'] < 2:
                as_loopback = True
                channels = 1

            self.__rate = self.__device['defaultSampleRate']
            stream = self.__p.open(
                format=self.__format,
                channels=channels,
                rate=self.__rate,
                input=True,
                frames_per_buffer=self.__chunk,
                input_device_index=self.__device["index"],
                as_loopback=as_loopback)

            while self.__recording:
                data = stream.read(self.__chunk)
                self.__q.put(data)

            stream.stop_stream()
            stream.close()

    def stop(self) -> None:
        self.__recording = False

    def __del__(self) -> None:
        self.__thread_running = False
        if self.__thread is not None:
            self.__thread.join()
        self.__p.terminate()
