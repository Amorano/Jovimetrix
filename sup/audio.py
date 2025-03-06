"""
Jovimetrix - Audio Support
"""

from enum import Enum

import numpy as np
from PIL import Image, ImageDraw

from .image import TYPE_PIXEL, \
    EnumImageType, \
    pil2cv

from .image.color import pixel_eval

from .image.adjust import EnumScaleMode, \
    image_scalefit

# ==============================================================================

class EnumGraphType(Enum):
    NORMAL = 0
    SOUNDCLOUD = 1

# ==============================================================================
# === VISUALIZE ===
# ==============================================================================

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
    color_line = pixel_eval(color_line, EnumImageType.BGR)
    color_back = pixel_eval(color_back, EnumImageType.BGR)
    image = Image.new('RGBA', (bar_count * line_width, height), color_line)
    draw = ImageDraw.Draw(image)
    for i, item in enumerate(max_array):
        item_height = item / line_ratio
        current_x = int((i + offset) * line_width)
        current_y = int((height - item_height) / 2)
        draw.line((current_x, current_y, current_x, current_y + item_height),
                fill=color_back, width=int(thickness * line_width))
    image = pil2cv(image)
    return image_scalefit(image, width, height, EnumScaleMode.FIT)
