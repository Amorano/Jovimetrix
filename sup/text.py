"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
TEXT support
"""

from enum import Enum
import textwrap

from matplotlib import font_manager
from PIL import Image, ImageFont, ImageDraw

from loguru import logger

from Jovimetrix.sup.image import pil2cv, TYPE_IMAGE, TYPE_PIXEL

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

# =============================================================================

def font_names() -> list[str]:
    mgr = font_manager.FontManager()
    return {font.name: font.fname for font in mgr.ttflist}

def text_size(draw:ImageDraw, text:str, font:ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height

def get_font_size(text, font, max_width=None, max_height=None):
    if max_width is None and max_height is None:
        raise ValueError('You need to pass max_width or max_height')

    font_size = 1
    text_size = ImageFont.truetype(font, font_size).getsize(text)
    if (max_width is not None and text_size[0] > max_width) or \
        (max_height is not None and text_size[1] > max_height):
        raise ValueError("Text can't be filled in only (%dpx, %dpx)" % \
                text_size)
    while True:
        if (max_width is not None and text_size[0] >= max_width) or \
            (max_height is not None and text_size[1] >= max_height):
            return font_size - 1
        font_size += 1
        text_size = ImageFont.truetype(font, font_size).getsize(text)

def text_draw(full_text: str, font: ImageFont, width:int, height:int, align:EnumAlignment, justify:EnumJustify, margin:int, line_spacing:int, color:TYPE_PIXEL) -> TYPE_IMAGE:

    img = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(img)

    def text_process_line(text: str, font: ImageFont) -> None:
        line_width, text_height = text_size(draw, text, font)
        x, y = 0, 0
        match justify:
            case EnumJustify.LEFT:
                x = margin
            case EnumJustify.RIGHT:
                x = width - line_width - margin
            case EnumJustify.CENTER:
                x = width / 2 - line_width / 2

        match align:
            case EnumAlignment.CENTER:
                y = height / 2 - text_height / 2
            case EnumAlignment.TOP:
                y = margin
            case EnumAlignment.BOTTOM:
                y = height - text_height - margin
        return x, y

    max_height = 0
    text = full_text.split('\n')
    for line in text:
        w, h = text_size(draw, line, font)
        max_height = max(max_height, h + line_spacing)

    y = 0
    for line in text:
        text_process_line(line, font, y)
        y += max_height
    return pil2cv(img)


"""
def wrap_text_and_calculate_height(self, text, font, max_width, line_height):
    wrapped_lines = []
    # Split the input text by newline characters to respect manual line breaks
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        words = paragraph.split()
        current_line = words[0] if words else ''

        for word in words[1:]:
            # Test if adding a new word exceeds the max width
            test_line = current_line + ' ' + word if current_line else word
            test_line_bbox = font.getbbox(test_line)
            w = test_line_bbox[2] - test_line_bbox[0]  # Right - Left for width
            if w <= max_width:
                current_line = test_line
            else:
                # If the current line plus the new word exceeds max width, wrap it
                wrapped_lines.append(current_line)
                current_line = word

        # Don't forget to add the last line of the paragraph
        wrapped_lines.append(current_line)

    # Calculate the total height considering the custom line height
    total_height = len(wrapped_lines) * line_height

    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text, total_height
"""

def text_autosize(text:str, font:str, width:int, height:int, columns:int=0) -> tuple[str, int, int, int]:
    img = Image.new("L", (width, height))
    draw = ImageDraw.Draw(img)
    if columns != 0:
        text = text.split('\n')
        lines = []
        for x in text:
            line = textwrap.wrap(x, columns, break_long_words=False)
            lines.extend(line)
        text = '\n'.join(lines)

    size = (1, 1, 1)
    font_size = 1
    while 1:
        ttf = ImageFont.truetype(font, font_size)
        w, h = text_size(draw, text, ttf)
        if w >= width or h >= height:
            break
        size = (font_size, w, h)
        font_size += 1
    return text, *size

def text_draw(full_text: str, font: ImageFont, width: int, height: int,
              align: EnumAlignment=EnumAlignment.CENTER,
              justify: EnumJustify=EnumJustify.CENTER,
              margin: int=0, line_spacing: int=0,
              color: TYPE_PIXEL=(255,255,255,255)) -> TYPE_IMAGE:

    img = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(img)
    text_lines = full_text.split('\n')
    count = len(text_lines)
    height_max = text_size(draw, full_text, font)[1] + line_spacing * count
    height_delta = height_max / count
    if align == EnumAlignment.TOP:
        y = margin
    elif align == EnumAlignment.BOTTOM:
        y = height - height_max - margin - line_spacing
    else:
        y = (height - height_max) / 2
    y = min(height, max(0, y))

    for line in text_lines:
        line_width = text_size(draw, line, font)[0]
        if justify == EnumJustify.LEFT:
            x = margin
        elif justify == EnumJustify.RIGHT:
            x = width - line_width - margin
        else:
            x = (width - line_width) / 2
        x = min(width, max(0, x))
        draw.text((x, y), line, fill=color, font=font)
        y += height_delta
    return pil2cv(img)
