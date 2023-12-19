"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Audio
"""

import torch
# import ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from Jovimetrix import deep_merge_dict, cv2mask, pil2cv, cv2tensor, \
    Logger, JOVImageBaseNode, Lexicon, \
    IT_REQUIRED, IT_WH, IT_RGB, IT_RGB_BACK

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

class GraphWaveNode(JOVImageBaseNode):
    NAME = "GRAPH WAVE (JOV) 🎶"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/AUDIO"
    RETURN_TYPES = ("IMAGE", "MASK", "WAVE")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.MASK, Lexicon.WAVE )
    OUTPUT_IS_LIST = (False, False, True)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.FILEN: ("STRING", {"default": ""}),
                Lexicon.AMT: ("INT", {"default": 100, "min": 32, "max": 8192, "step": 1})
            }}
        return deep_merge_dict(IT_REQUIRED, IT_WH, d, IT_RGB, IT_RGB_BACK)

    # #️⃣ 🪄
    def __init__(self) -> None:
        self.__filen = None
        self.__data = None

    def run(self, filen: str, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        width = kw.get(Lexicon.WIDTH, None)
        height = kw.get(Lexicon.HEIGHT, None)
        bars = kw.get(Lexicon.AMT, None)
        rgb = kw.get(Lexicon.RGB, None)
        back = kw.get(Lexicon.RGB_BACK, None)


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
            image = graph_sausage(self.__data, bars, width, height, rgb, back)

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