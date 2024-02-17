"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
TEXT support
"""

from enum import Enum
from typing import Any

import matplotlib.font_manager
from PIL import ImageFont, ImageDraw

from loguru import logger

# =============================================================================

class EnumShapes(Enum):
    CIRCLE = 0
    SQUARE = 1
    ELLIPSE = 2
    RECTANGLE = 3
    POLYGON = 4

class EnumAlignment(Enum):
    TOP = 10
    CENTER = 0
    BOTTOM = 20

class EnumJustify(Enum):
    LEFT = 10
    CENTER = 0
    RIGHT = 20
    JUSTIFY = 30

# =============================================================================
def font_all() -> dict[str, str]:
    mgr = matplotlib.font_manager.FontManager()
    return {font.name: font.fname for font in mgr.ttflist}

def font_all_names() -> list[str]:
    return sorted(font_all().keys())

def text_align(align:EnumAlignment, height:int, text_height:int, margin:int) -> Any:
    match align:
        case EnumAlignment.CENTER:
            return height / 2 - text_height / 2
        case EnumAlignment.TOP:
            return margin
        case EnumAlignment.BOTTOM:
            return height - text_height - margin

def text_justify(justify:EnumJustify, width:int, line_width:int, margin:int) -> Any:
    x = 0
    match justify:
        case EnumJustify.LEFT:
            x = margin
        case EnumJustify.RIGHT:
            x = width - line_width - margin
        case EnumJustify.CENTER:
            x = width/2 - line_width/2
    return x

def text_size(draw:ImageDraw, text:str, font:ImageFont) -> tuple:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height

def get_font_size(text, font, max_width=None, max_height=None):
    if max_width is None and max_height is None:
        raise ValueError('You need to pass max_width or max_height')

    font_size = 1
    text_size = get_text_size(font, font_size, text)
    if (max_width is not None and text_size[0] > max_width) or \
        (max_height is not None and text_size[1] > max_height):
        raise ValueError("Text can't be filled in only (%dpx, %dpx)" % \
                text_size)
    while True:
        if (max_width is not None and text_size[0] >= max_width) or \
            (max_height is not None and text_size[1] >= max_height):
            return font_size - 1
        font_size += 1
        text_size = get_text_size(font, font_size, text)

def write_text(xy, text, font_filename, font_size=11,
                color=(0, 0, 0), max_width=None, max_height=None):
    x, y = xy
    if font_size == 0 and \
        (max_width is not None or max_height is not None):
        font_size = get_font_size(text, font_filename, max_width,
                                        max_height)
    text_size = get_text_size(font_filename, font_size, text)
    font = ImageFont.truetype(font_filename, font_size)
    if x == 'center':
        x = (self.size[0] - text_size[0]) / 2
    if y == 'center':
        y = (self.size[1] - text_size[1]) / 2
    draw.text((x, y), text, font=font, fill=color)
    return text_size

def get_text_size(font_filename, font_size, text):
    return ImageFont.truetype(font_filename, font_size).getsize(text)

def write_text_box(width, height, xy, text, box_width, font_filename,
                    font_size=11, color=(0, 0, 0), justify:EnumJustify=EnumJustify.LEFT,
                    justify_last_line=False, position:EnumAlignment=EnumAlignment.TOP,
                    line_spacing=1.0):
    x, y = xy
    lines = []
    line = []
    words = text.split()
    for word in words:
        new_line = ' '.join(line + [word])
        size = get_text_size(font_filename, font_size, new_line)
        text_height = size[1] * line_spacing
        last_line_bleed = text_height - size[1]
        if size[0] <= box_width:
            line.append(word)
        else:
            lines.append(line)
            line = [word]
    if line:
        lines.append(line)
    lines = [' '.join(line) for line in lines if line]

    if position == EnumAlignment.CENTER:
        height = (height - len(lines)*text_height + last_line_bleed)/2
    elif position == EnumAlignment.BOTTOM:
        height = height - len(lines)*text_height + last_line_bleed
    else:
        height = y
    height -= text_height  # the loop below will fix this height

    for index, line in enumerate(lines):
        if justify == EnumJustify.LEFT:
            write_text((x, height), line, font_filename, font_size, color)

        elif justify == EnumJustify.RIGHT:
            total_size = get_text_size(font_filename, font_size, line)
            x_left = x + box_width - total_size[0]
            write_text((x_left, height), line, font_filename, font_size, color)

        elif justify == EnumJustify.CENTER:
            total_size = get_text_size(font_filename, font_size, line)
            x_left = int(x + ((box_width - total_size[0]) / 2))
            write_text((x_left, height), line, font_filename, font_size, color)

        elif justify == EnumJustify.JUSTIFY:
            words = line.split()
            if (index == len(lines) - 1 and not justify_last_line) or \
                len(words) == 1:
                write_text((x, height), line, font_filename, font_size, color)
                continue

            line_without_spaces = ''.join(words)
            total_size = get_text_size(font_filename, font_size, line_without_spaces)
            space_width = (box_width - total_size[0]) / (len(words) - 1.0)
            start_x = x
            for word in words[:-1]:
                write_text((start_x, height), word, font_filename, font_size, color)
                word_size = get_text_size(font_filename, font_size, word)
                start_x += word_size[0] + space_width

            last_word_size = get_text_size(font_filename, font_size, words[-1])
            last_word_x = x + box_width - last_word_size[0]
            write_text((last_word_x, height), words[-1], font_filename, font_size, color)

        height += text_height

    return (box_width, height - y)
