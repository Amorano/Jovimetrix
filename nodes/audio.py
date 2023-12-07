"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Audio
"""

import torch
import ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from Jovimetrix import Logger, cv2mask, pil2cv, cv2tensor, \
    JOVImageBaseNode, MIN_HEIGHT

# =============================================================================
# === LOADERS ===
# =============================================================================

def load_audio(file_path) -> np.ndarray[np.int16]:
    cmd = (
        ffmpeg.input(file_path)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1)
        .run(input=None, capture_stdout=False, capture_stderr=False)
    )
    Logger.debug("load_audio", file_path)
    return np.frombuffer(cmd[0], dtype=np.int16)

# =============================================================================
# === EXTRACT ===
# =============================================================================

def extract_wave(data: np.ndarray) -> np.ndarray[np.float32]:
    # Normalize audio data to the range [-1, 1]
    return data.astype(np.float32) / 32767.0

# =============================================================================
# === VISUALIZE ===
# =============================================================================

def graph_sausage(data: np.ndarray, bar_count:int, width:int, height:int,
                    color_line:tuple[float, float, float]=(0.7,0.7,0.7),
                    color_back:tuple[float, float, float]=(0.,0.,0.)) -> np.ndarray[np.int8]:

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

    image = image.resize((width, height))
    Logger.debug(f"graph_sausage {bar_count} [{width}x{height}]")
    return pil2cv(image)

# =============================================================================

class GraphAudioNode(JOVImageBaseNode):
    NAME = "🎶 Graph Audio Wave (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/AUDIO"
    RETURN_TYPES = ("IMAGE", "MASK", "WAVE")
    RETURN_NAMES = ("🖼️", "😷", "〰️" )
    OUTPUT_IS_LIST = (False, False, True)

    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required":{
                "filen": ("STRING", {"default": ""})},
            "optional": {
                "bars": ("INT", {"default": 100, "min": 32, "max": 8192, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 1}),
                "height": ("INT", {"default": MIN_HEIGHT, "min": 32, "max": 8192, "step": 1}),
                "barR": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "barG": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "barB": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "backR": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "backG": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "backB": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
        }}

    def __init__(self) -> None:
        self.__filen = None
        self.__data = None

    def run(self, filen: str, bars:int, width: int, height: int,
            barR: float, barG: float, barB: float,
            backR: float, backG: float, backB: float ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.__filen != filen:
            self.__data = None
            try:
                self.__data = load_audio(filen)
                self.__filen = filen
            except ffmpeg._run.Error as _:
                pass
            except Exception as e:
                Logger.err(str(e))

        image = np.zeros((1, 1), dtype=np.int16)
        if self.__data is not None:
            image = graph_sausage(self.__data, bars, width, height, (barR, barG, barB), (backR, backG, backB))

        image = cv2tensor(image)
        mask = cv2mask(image)
        #mask = torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0)
        #image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

        data = extract_wave(self.__data)
        return (image, mask, data,)

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    data = load_audio('./res/aud_mp3.mp3')
    graph_sausage(data, 50, 512, 1014, (169, 171, 172), (0, 0, 0))