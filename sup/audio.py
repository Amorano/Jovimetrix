"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix

                    Copyright 2023 Alexander Morano (Joviex)

Audio Supports.
"""

from math import e
from re import X
import ffmpeg
import numpy as np
from PIL import Image, ImageDraw

try:
    from .util import loginfo, logwarn, logerr, logdebug
except:
    from sup.util import loginfo, logwarn, logerr, logdebug

# =============================================================================
# === LOADERS ===
# =============================================================================

def load_audio(file_path) -> np.ndarray[np.int16]:
    cmd = (
        ffmpeg.input(file_path)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1)
        .run(input=None, capture_stdout=True, capture_stderr=True)
    )
    logdebug(f"[load_audio] {file_path}")
    return np.frombuffer(cmd[0], dtype=np.int16)

# =============================================================================
# === EXTRACT ===
# =============================================================================

def wave(data: np.ndarray) -> np.ndarray[np.float32]:
    # Normalize audio data to the range [-1, 1]
    return data.astype(np.float32) / 32767.0

# =============================================================================
# === VISUALIZE ===
# =============================================================================

def graph_sausage(data: np.ndarray, bar_count:int, width:int, height:int,
                    color_line:tuple[float, float, float]=(0.7,0.7,0.7),
                    color_back:tuple[float, float, float]=(0.,0.,0.)) -> Image:

    # Normalize audio data to the range [-1, 1]
    normalized_data = data.astype(np.float32) / 32767.0

    length = len(normalized_data)
    ratio = length / bar_count
    count = 0
    maximum_item = 0
    max_array = []
    highest_line = 0

    for d in normalized_data:
        if count < ratio:
            count += 1
            if abs(d) > maximum_item:
                maximum_item = abs(d)
            continue

        max_array.append(maximum_item)
        if maximum_item > highest_line:
            highest_line = maximum_item

        maximum_item = 0
        count = 1

    line_width = (width + bar_count) // bar_count
    line_ratio = highest_line / height

    color_back = tuple([int(255*x) for x in color_back])
    color_line = tuple([int(255*x) for x in color_line])

    image = Image.new('RGB', (bar_count * line_width, height), color_back)
    draw = ImageDraw.Draw(image)

    current_x = 1
    for item in max_array:
        item_height = item / line_ratio
        current_y = (height - item_height) / 2
        draw.line((current_x, current_y, current_x, current_y + item_height), fill=color_line, width=4)
        current_x = current_x + line_width

    logdebug(f"[graph_sausage] {bar_count} [{width}x{height}]")
    return image.resize((width, height))

# =============================================================================
# === TESTING! ===
# =============================================================================

if __name__ == "__main__":
    data = load_audio('./res/aud_wav.wav')
    graph_sausage(data, 50, 512, 1014, (169, 171, 172), (0, 0, 0))
    data = load_audio('./res/aud_mp3.mp3')
    graph_sausage(data, 50, 512, 1014, (169, 171, 172), (0, 0, 0))