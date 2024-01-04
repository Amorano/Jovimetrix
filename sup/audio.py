"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Audio Support
"""

import io
from enum import Enum
from urllib.request import urlopen

import librosa
import ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from Jovimetrix import TYPE_PIXEL

# =============================================================================

class EnumGraphType(Enum):
    NORMAL = 0
    SOUNDCLOUD = 1

# =============================================================================
# === LOADERS ===
# =============================================================================

def load_audio(file_path) -> np.ndarray[np.int16]:
    cmd = (
        ffmpeg.input(file_path)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1)
        .run(input=None, capture_stdout=False, capture_stderr=False)
    )
    # Logger.debug("load_audio", file_path)
    return np.frombuffer(cmd[0], dtype=np.int16)

def load_audio(url: str) -> tuple[np.ndarray[np.int16], float]:
    if url.startswith("http"):
        url = io.BytesIO(urlopen(url).read())
    # data, rate = sf.read(url, dtype='float32')
    data, rate = librosa.load(url)
    # data = librosa.resample(data.T, orig_sr=rate, target_sr=22050)
    return data, rate

# =============================================================================
# === EXTRACT ===
# =============================================================================

def wave_extract(data: np.ndarray) -> np.ndarray[np.float32]:
    # Normalize audio data to the range [-1, 1]
    return data.astype(np.float32) / 32767.0

# =============================================================================
# === VISUALIZE ===
# =============================================================================

def graph_sausage(data: np.ndarray, bar_count:int, width:int, height:int,
                    thickness: float = 0.5, offset: float = 0.0,
                    color_line:TYPE_PIXEL=(172, 172, 172),
                    color_back:TYPE_PIXEL=(0, 0, 0)) -> np.ndarray[np.int8]:

    # Normalize [-1, 1]
    normalized_data = data.astype(np.float32) / 32767.0
    length = len(normalized_data)
    ratio = length / bar_count

    # Vectorize
    max_array = np.maximum.reduceat(np.abs(normalized_data), np.arange(0, length, ratio, dtype=int))
    highest_line = max_array.max()
    line_width = (width + bar_count) // bar_count
    line_ratio = highest_line / height

    image = Image.new('RGB', (bar_count * line_width, height), color_back)
    draw = ImageDraw.Draw(image)

    for i, item in enumerate(max_array):
        item_height = item / line_ratio
        current_x = int((i + offset) * line_width)
        current_y = int((height - item_height) / 2)
        draw.line((current_x, current_y, current_x, current_y + item_height),
                  fill=color_line, width=int(thickness * line_width))

    return image

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    print(sf.__libsndfile_version__)
    url = './_res/aud.wav'
    url = "https://upload.wikimedia.org/wikipedia/commons/b/bb/Test_ogg_mp3_48kbps.wav"
    url = "http://tinyurl.com/shepard-risset"
    data, rate = load_audio(url)
    img = graph_sausage(data, 65, 1024, 512, color_back=(0, 48, 0))
    img.save('./_res/tst/sausage_graph.png')
