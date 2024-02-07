"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from enum import Enum
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager
from loguru import logger

import comfy
from server import PromptServer

from Jovimetrix import JOVImageSimple, JOVImageMultiple, \
    IT_DEPTH, IT_PIXEL, IT_RGBA, IT_WH, IT_SCALE, IT_ROT, IT_INVERT, \
    IT_REQUIRED, MIN_IMAGE_SIZE

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import deep_merge_dict, parse_tuple, parse_number, \
    zip_longest_fill, EnumTupleType

from Jovimetrix.sup.image import image_rotate, channel_add, image_stereogram, \
    pil2tensor, pil2cv, cv2tensor, cv2mask, tensor2cv, shape_ellipse, shape_polygon, \
    shape_quad, image_invert, \
    EnumEdge, IT_EDGE

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
    CENTER=0
    TOP=1
    BOTTOM=2

class EnumJustify(Enum):
    CENTER=0
    LEFT=1
    RIGHT=2

# =============================================================================

def text_align(align:EnumAlignment, height:int, text_height:int, margin:int) -> Any:
    y = 0
    match align:
        case EnumAlignment.CENTER:
            y = height / 2 - text_height / 2
        case EnumAlignment.TOP:
            y = margin
        case EnumAlignment.BOTTOM:
            y = height - text_height - margin
    return y

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

def text_size(draw:ImageDraw, text:str, font) -> tuple:
    bbox = draw.textbbox((0, 0), text, font=font)

    # Calculate the text width and height
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height

# =============================================================================

class ConstantNode(JOVImageMultiple):
    NAME = "CONSTANT (JOV) 🟪"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_WH, IT_RGBA)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        color = parse_tuple(Lexicon.RGB_A, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)
        params = [tuple(x) for x in zip_longest_fill(wihi, color)]
        images = []
        masks = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (wihi, color) in enumerate(params):
            width, height = wihi
            image = Image.new("RGB", (width, height), color[:3])
            mask = Image.new("L", (width, height), color[3])
            images.append(pil2tensor(image))
            masks.append(pil2tensor(mask))
            pbar.update_absolute(idx)

        return (images, masks,)

class ShapeNode(JOVImageMultiple):
    NAME = "SHAPE GENERATOR (JOV) ✨"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.SHAPE: (EnumShapes._member_names_, {"default": EnumShapes.CIRCLE.name}),
            Lexicon.SIDES: ("INT", {"default": 3, "min": 3, "max": 100, "step": 1}),
            Lexicon.RGB: ("VEC3", {"default": (255, 255, 255), "min": 0, "max": 255, "step": 1, "label":
                                    [Lexicon.R, Lexicon.G, Lexicon.B]}),
            Lexicon.RGB_B: ("VEC3", {"default": (0, 0, 0), "min": 0, "max": 255, "step": 1, "label":
                                       [Lexicon.R, Lexicon.G, Lexicon.B]})
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_ROT, IT_SCALE, IT_EDGE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        shape = kw.get(Lexicon.SHAPE, EnumShapes.CIRCLE)
        sides = kw.get(Lexicon.SIDES, 3)
        angle = kw.get(Lexicon.ANGLE, 0)
        edge = kw.get(Lexicon.EDGE, EnumEdge.CLIP)
        size = parse_tuple(Lexicon.SIZE, kw, EnumTupleType.FLOAT, default=(1., 1.,))
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        color = parse_tuple(Lexicon.RGB, kw, default=(255, 255, 255))
        bgcolor = parse_tuple(Lexicon.RGB_B, kw, default=(0, 0, 0))
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1])
        params = [tuple(x) for x in zip_longest_fill(shape, sides, angle, edge, size, wihi, color, bgcolor, i)]
        images = []
        masks = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (shape, sides, angle, edge, size, wihi, color, bgcolor, i) in enumerate(params):
            width, height = wihi
            sizeX, sizeY = size
            edge = EnumEdge[edge]
            shape = EnumShapes[shape]
            match shape:
                case EnumShapes.SQUARE:
                    img = shape_quad(width, height, sizeX, sizeX, fill=color, back=bgcolor)
                    mask = shape_quad(width, height, sizeX, sizeX)

                case EnumShapes.ELLIPSE:
                    img = shape_ellipse(width, height, sizeX, sizeY, fill=color, back=bgcolor)
                    mask = shape_ellipse(width, height, sizeX, sizeY)

                case EnumShapes.RECTANGLE:
                    img = shape_quad(width, height, sizeX, sizeY, fill=color, back=bgcolor)
                    mask = shape_quad(width, height, sizeX, sizeY)

                case EnumShapes.POLYGON:
                    img = shape_polygon(width, height, sizeX, sides, fill=color, back=bgcolor)
                    mask = shape_polygon(width, height, sizeX, sides)

                case EnumShapes.CIRCLE:
                    img = shape_ellipse(width, height, sizeX, sizeX, fill=color, back=bgcolor)
                    mask = shape_ellipse(width, height, sizeX, sizeX)

            img = pil2cv(img)
            mask = pil2cv(mask)
            img = image_rotate(img, angle, edge=edge)
            mask = image_rotate(mask, angle, edge=edge)
            if i != 0:
                img = image_invert(img, i)

            if img.shape[2] == 3:
                img = channel_add(img, 0)

            images.append(cv2tensor(img[:,:,:3]))
            masks.append(cv2mask(mask[:, :, 0]))
            pbar.update_absolute(idx)

        return (images, masks,)

class TextNode(JOVImageMultiple):
    NAME = "TEXT GENERATOR (JOV) 📝"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.STRING: ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            Lexicon.FONT: (FONT_NAMES, {"default": FONT_NAMES[0]}),
            Lexicon.FONT_SIZE: ("INT", {"default": 100, "min": 1, "step": 1}),
            Lexicon.RGB: ("VEC3", {"default": (255, 255, 255), "min": 0, "max": 255, "step": 1, "label":
                                    [Lexicon.R, Lexicon.G, Lexicon.B]}),
            Lexicon.RGB_B: ("VEC3", {"default": (0, 0, 0), "min": 0, "max": 255, "step": 1, "label":
                                    [Lexicon.R, Lexicon.G, Lexicon.B]}),
            Lexicon.LETTER: ("BOOLEAN", {"default": False}),
            Lexicon.ALIGN: (EnumAlignment._member_names_, {"default": EnumAlignment.CENTER.name}),
            Lexicon.JUSTIFY: (EnumJustify._member_names_, {"default": EnumJustify.CENTER.name}),
            Lexicon.MARGIN: ("INT", {"default": 0, "min": -1024, "max": 1024}),
            Lexicon.SPACING: ("INT", {"default": 25, "min": -1024, "max": 1024}),
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_ROT, IT_EDGE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        if len(full_text := kw.get(Lexicon.STRING, [""])) == 0:
            full_text = [""]

        #if not type(full_text) == list:
        #   full_text = [full_text]
        # logger.debug(full_text)

        font = kw.get(Lexicon.FONT, FONT_NAMES[0])
        size = kw.get(Lexicon.FONT_SIZE, [100])
        color = parse_tuple(Lexicon.RGB, kw, default=(255, 255, 255))
        bgcolor = parse_tuple(Lexicon.RGB_B, kw, default=(0, 0, 0))
        align = kw.get(Lexicon.ALIGN, [EnumAlignment.CENTER])
        justify = kw.get(Lexicon.JUSTIFY, [EnumJustify.CENTER])
        margin = kw.get(Lexicon.MARGIN, [0])
        line_spacing = kw.get(Lexicon.SPACING, [25])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        angle = parse_number(Lexicon.ANGLE, kw, EnumTupleType.FLOAT, [0])
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [0])
        letter = kw.get(Lexicon.LETTER, [False])
        params = [tuple(x) for x in zip_longest_fill(full_text, font, size, color, bgcolor, align, justify, margin, line_spacing, wihi, angle, edge, i, letter)]

        images = []
        masks = []

        def process(img, mask, ang, e, invert) -> None:
            img = pil2cv(img)
            mask = pil2cv(mask)
            img = image_rotate(img, ang, edge=e)
            mask = image_rotate(mask, ang, edge=e)

            if invert != 0:
                img = image_invert(img, invert)

            images.append(cv2tensor(img))
            masks.append(cv2mask(mask))

        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (full_text, font, size, color, bgcolor, align, justify, margin, line_spacing, wihi, angle, edge, i, letter) in enumerate(params):
            font = FONTS[font]
            font = ImageFont.truetype(font, size)
            align = EnumAlignment[align]
            justify = EnumJustify[justify]
            edge = EnumEdge[edge]
            width, height = wihi

            # if we should output single letters instead of full phrase
            if not letter:
                img = Image.new("RGB", (width, height), bgcolor)
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(img)
                draw_mask = ImageDraw.Draw(mask)

                max_width = 0
                max_height = 0
                text = full_text.split('\n')
                for line in text:
                    w, h = text_size(draw, line, font)
                    max_width = max(max_width, w)
                    max_height = max(max_height, h + line_spacing)

                y = 0
                text_height = max_height * len(text)
                for idx, line in enumerate(text):
                    # Calculate the width of the current line
                    line_width, _ = text_size(draw, line, font)

                    # Get the text x and y positions for each line
                    x = text_justify(justify, width, line_width, margin)
                    y = text_align(align, height, text_height, margin)
                    y += (idx * max_height)

                    # Add the current line to the text mask
                    draw.text((x, y), line, fill=color, font=font)
                    draw_mask.text((x, y), line, fill=255, font=font)

                process(img, mask, angle, edge, i)

            else:
                text = full_text.replace('\n', '')
                for idx, letter in enumerate(text):
                    img = Image.new("RGB", (width, height), bgcolor)
                    mask = Image.new("L", (width, height), 0)
                    draw = ImageDraw.Draw(img)
                    draw_mask = ImageDraw.Draw(mask)
                    x = text_justify(justify, width, line_width, margin)
                    y = text_align(align, height, text_height, margin)
                    draw.text((x, y), line, fill=color, font=font)
                    draw_mask.text((x, y), line, fill=255, font=font)
                    process(img, mask, angle, edge, i)

            pbar.update_absolute(idx)

        return (images, masks, )

class StereogramNode(JOVImageSimple):
    NAME = "STEREOGRAM (JOV) 📻"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = "Make a magic eye stereogram."
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
