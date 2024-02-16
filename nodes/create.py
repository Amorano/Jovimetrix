"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import math
import textwrap
from typing import Any
from enum import Enum
from itertools import zip_longest

import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager
from loguru import logger

import comfy
from server import PromptServer

from Jovimetrix import JOVImageSimple, JOVImageMultiple, \
    IT_RGB_B, IT_RGBA_A, \
    IT_DEPTH, IT_PIXEL, IT_WH, IT_SCALE, IT_ROT, IT_INVERT, \
    IT_REQUIRED, MIN_IMAGE_SIZE

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import deep_merge_dict, parse_tuple, \
    parse_number, zip_longest_fill, EnumTupleType

from Jovimetrix.sup.image import channel_solid, cv2tensor_full,  \
    image_mask_add, image_matte, image_rotate, \
    image_stereogram, pil2cv, cv2tensor, cv2mask, \
    pixel_eval, tensor2cv, shape_ellipse, shape_polygon, \
    shape_quad, image_invert, \
    EnumEdge, EnumImageType, \
    IT_EDGE

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
FONT_MANAGER = matplotlib.font_manager.FontManager()
FONTS = {font.name: font.fname for font in FONT_MANAGER.ttflist}
FONT_NAMES = sorted(FONTS.keys())

# =============================================================================

class EnumShapes(Enum):
    CIRCLE=0
    SQUARE=1
    ELLIPSE=2
    RECTANGLE=3
    POLYGON=4

class EnumAlignment(Enum):
    TOP=1
    CENTER=0
    BOTTOM=2

class EnumJustify(Enum):
    LEFT=1
    CENTER=0
    RIGHT=2

# =============================================================================

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

# =============================================================================

class ConstantNode(JOVImageMultiple):
    NAME = "CONSTANT (JOV) 🟪"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Create a single RGBA block of color. Useful for masks, overlays and general filtering."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_WH, IT_RGBA_A)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        color = parse_tuple(Lexicon.RGBA_A, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)
        images = []
        params = [tuple(x) for x in zip_longest_fill(wihi, color)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (wihi, color) in enumerate(params):
            width, height = wihi
            color = pixel_eval(color, EnumImageType.BGRA)
            img = channel_solid(width, height, color, chan=EnumImageType.BGRA)
            logger.debug(img.shape)
            img = cv2tensor_full(img)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))

class ShapeNode(JOVImageMultiple):
    NAME = "SHAPE GENERATOR (JOV) ✨"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Generate polyhedra for masking or texture work."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.SHAPE: (EnumShapes._member_names_, {"default": EnumShapes.CIRCLE.name}),
            Lexicon.SIDES: ("INT", {"default": 3, "min": 3, "max": 100, "step": 1}),
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_RGBA_A, IT_RGB_B, IT_WH, IT_ROT, IT_SCALE, IT_EDGE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        shape = kw.get(Lexicon.SHAPE, EnumShapes.CIRCLE)
        sides = kw.get(Lexicon.SIDES, 3)
        angle = kw.get(Lexicon.ANGLE, 0)
        edge = kw.get(Lexicon.EDGE, EnumEdge.CLIP)
        size = parse_tuple(Lexicon.SIZE, kw, EnumTupleType.FLOAT, default=(1., 1.,))
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        color = parse_tuple(Lexicon.RGBA_A, kw, default=(255, 255, 255, 255))
        bgcolor = parse_tuple(Lexicon.RGB_B, kw, default=(0, 0, 0))
        invert = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1])
        params = [tuple(x) for x in zip_longest_fill(shape, sides, angle, edge, size, wihi, color, bgcolor, invert)]
        images = []
        params = [tuple(x) for x in zip_longest_fill(wihi, color)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (shape, sides, angle, edge, size, wihi, color, bgcolor, invert) in enumerate(params):
            width, height = wihi
            sizeX, sizeY = size
            edge = EnumEdge[edge]
            shape = EnumShapes[shape]
            match shape:
                case EnumShapes.SQUARE:
                    img = shape_quad(width, height, sizeX, sizeX, fill=color, back=bgcolor)
                    mask = shape_quad(width, height, sizeX, sizeX, fill=color[3])

                case EnumShapes.ELLIPSE:
                    img = shape_ellipse(width, height, sizeX, sizeY, fill=color, back=bgcolor)
                    mask = shape_ellipse(width, height, sizeX, sizeY, fill=color[3])

                case EnumShapes.RECTANGLE:
                    img = shape_quad(width, height, sizeX, sizeY, fill=color, back=bgcolor)
                    mask = shape_quad(width, height, sizeX, sizeY, fill=color[3])

                case EnumShapes.POLYGON:
                    img = shape_polygon(width, height, sizeX, sides, fill=color, back=bgcolor)
                    mask = shape_polygon(width, height, sizeX, sides, fill=color[3])

                case EnumShapes.CIRCLE:
                    img = shape_ellipse(width, height, sizeX, sizeX, fill=color, back=bgcolor)
                    mask = shape_ellipse(width, height, sizeX, sizeX, fill=color[3])

            img = pil2cv(img)
            if invert != 0:
                img = image_invert(img, invert)
            mask = pil2cv(mask)[:,:,0]
            img = image_mask_add(img, mask)
            img = image_rotate(img, angle, edge=edge)
            img = cv2tensor_full(img)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))

class TextNode(JOVImageMultiple):
    NAME = "TEXT GENERATOR (JOV) 📝"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Use any system font with auto-fit or manual placement."
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.STRING: ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            Lexicon.FONT: (FONT_NAMES, {"default": FONT_NAMES[0]}),
            Lexicon.LETTER: ("BOOLEAN", {"default": False}),
            Lexicon.AUTOSIZE: ("BOOLEAN", {"default": False}),
            Lexicon.RGBA_A: ("VEC3", {"default": (255, 255, 255, 255), "min": 0, "max": 255, "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True}),
            Lexicon.MATTE: ("VEC3", {"default": (0, 0, 0), "min": 0, "max": 255, "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B], "rgb": True}),
            #
            Lexicon.COLUMNS: ("INT", {"default": 0, "min": 0, "step": 1}),
            # if auto on, hide these...
            Lexicon.FONT_SIZE: ("INT", {"default": 100, "min": 1, "step": 1}),
            Lexicon.ALIGN: (EnumAlignment._member_names_, {"default": EnumAlignment.CENTER.name}),
            Lexicon.JUSTIFY: (EnumJustify._member_names_, {"default": EnumJustify.CENTER.name}),
            Lexicon.MARGIN: ("INT", {"default": 0, "min": -1024, "max": 1024}),
            Lexicon.SPACING: ("INT", {"default": 25, "min": -1024, "max": 1024}),
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_ROT, IT_EDGE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        if len(full_text := kw.get(Lexicon.STRING, [""])) == 0:
            full_text = [""]
        font_idx = kw.get(Lexicon.FONT, FONT_NAMES[0])
        autosize = kw.get(Lexicon.AUTOSIZE, [False])
        letter = kw.get(Lexicon.LETTER, [False])
        color = parse_tuple(Lexicon.RGBA_A, kw, default=(255, 255, 255, 255))
        bgcolor = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0))
        columns = kw.get(Lexicon.COLUMNS, [0])
        size = kw.get(Lexicon.FONT_SIZE, [100])
        align = kw.get(Lexicon.ALIGN, [EnumAlignment.CENTER])
        justify = kw.get(Lexicon.JUSTIFY, [EnumJustify.CENTER])
        margin = kw.get(Lexicon.MARGIN, [0])
        line_spacing = kw.get(Lexicon.SPACING, [25])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        angle = parse_number(Lexicon.ANGLE, kw, EnumTupleType.FLOAT, [0])
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        images = []
        params = [tuple(x) for x in zip_longest_fill(full_text, font_idx, autosize, letter, color, bgcolor, columns, size, align, justify, margin, line_spacing, wihi, angle, edge)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (full_text, font_idx, autosize, letter, color, bgcolor, columns, size, align, justify, margin, line_spacing, wihi, angle, edge) in enumerate(params):

            font_name = FONTS[font_idx]
            width, height = wihi
            align = EnumAlignment[align]
            justify = EnumJustify[justify]
            edge = EnumEdge[edge]

            def process(img, mask, idx) -> None:
                img = pil2cv(img)
                mask = pil2cv(mask)[:,:,0]
                img = image_mask_add(img, mask)
                img = image_rotate(img, angle, edge=edge)
                matte = pixel_eval(bgcolor)
                img = cv2tensor_full(img, matte)
                images.append(img)
                pbar.update_absolute(idx)

            def process_line(text: str, font: ImageFont, draw:ImageDraw, mask:ImageDraw, y:int=0, auto_align:bool=True) -> None:
                # Calculate the width of the current line
                line_width, text_height = text_size(draw, text, font)
                # Get the text x and y positions for each line
                x = text_justify(justify, width, line_width, margin)
                if auto_align:
                    y += text_align(align, height, text_height, margin)
                # Add the current line to the text mask
                draw.text((x, y), text, fill=color[:3], font=font)
                mask.text((x, y), text, fill=color[3], font=font)

            def process_block(text: str, font: ImageFont, draw:ImageDraw, mask:ImageDraw, max_width:int, max_height:int) -> None:
                y = 0
                for line in text:
                    process_line(line, font, draw, mask, y=y)
                    y += max_height

            if letter:
                text = full_text.replace('\n', '')
                font = ImageFont.truetype(font_name, size)
                for ch in text:
                    img = Image.new("RGB", (width, height), bgcolor)
                    mask = Image.new("L", (width, height), 0)
                    draw = ImageDraw.Draw(img)
                    draw_mask = ImageDraw.Draw(mask)
                    process_line(str(ch), font, draw, draw_mask)
                    process(img, mask, idx)
                continue

            # full text auto-fit mode
            if autosize:
                img = Image.new("RGB", (width, height), bgcolor)
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(img)
                draw_mask = ImageDraw.Draw(mask)
                if columns == 0:
                    side = math.sqrt(len(full_text))
                    columns = int(math.ceil(side))

                pre_text = full_text.split('\n')
                lines = []
                for x in pre_text:
                    line = textwrap.wrap(x, columns, break_long_words=False)
                    lines.extend(line)

                line_count = len(lines)
                all_line_height = line_spacing*line_count
                line = max(lines, key=len)
                font_size, max_width, max_height = 1, 0, 0
                while 1:
                    font = ImageFont.truetype(font_name, font_size)
                    w, h = text_size(draw, line, font)
                    if (w+margin*2) >= width or h >= (height+all_line_height):
                        break
                    max_width, max_height = w, h
                    font_size += 1

                y = margin
                match align:
                    case EnumAlignment.CENTER:
                        y = height / 2 - (max_height+line_spacing) * line_count / 2
                    case EnumAlignment.BOTTOM:
                        y = height - (max_height+line_spacing) * line_count

                logger.debug([font_size, y, all_line_height, max_height, line_count])
                for line in lines:
                    process_line(line, font, draw, draw_mask, y, False)
                    y += (max_height + line_spacing)
            else:
                img = Image.new("RGB", (width, height), bgcolor)
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(img)
                draw_mask = ImageDraw.Draw(mask)
                max_width = 0
                max_height = 0
                font = ImageFont.truetype(font_name, size)
                text = full_text.split('\n')
                for line in text:
                    w, h = text_size(draw, line, font)
                    max_width = max(max_width, w)
                    max_height = max(max_height, h + line_spacing)
                process_block(text, font, draw, draw_mask, max_width, max_height)
            process(img, mask, idx)
        return list(zip(*images))

class StereogramNode(JOVImageSimple):
    NAME = "STEREOGRAM (JOV) 📻"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Make a magic eye stereograms."
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.TILE: ("INT", {"default": 8, "min": 1}),
                Lexicon.NOISE: ("FLOAT", {"default": 0.33, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.GAMMA: ("FLOAT", {"default": 0.33, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.SHIFT: ("FLOAT", {"default": 1., "min": -1, "max": 1, "step": 0.01}),
        }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL, IT_DEPTH, d)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        img = kw.get(Lexicon.PIXEL, [None])
        depth = kw.get(Lexicon.DEPTH, [None])
        divisions = kw.get(Lexicon.TILE, [8])
        noise = kw.get(Lexicon.NOISE, [0.33])
        gamma = kw.get(Lexicon.VALUE, [0.33])
        shift = kw.get(Lexicon.SHIFT, [1])
        params = [tuple(x) for x in zip_longest_fill(img, depth, divisions, noise, gamma, shift)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, depth, divisions, noise, gamma, shift) in enumerate(params):
            if img is None:
                empty = torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), device='cpu')
                images.append(empty)
                continue

            img = tensor2cv(img)
            depth = tensor2cv(depth)
            img = image_stereogram(img, depth, divisions, noise, gamma, shift)
            images.append(cv2tensor(img))
            pbar.update_absolute(idx)

        return (images, )
