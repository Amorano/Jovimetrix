"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from re import X
from typing import Any

import torch
from PIL import ImageFont

from loguru import logger

import comfy
# from server import PromptServer

from Jovimetrix import IT_TRANS, JOVImageSimple, JOVImageMultiple, \
    JOV_HELP_URL, IT_RGBA_A, IT_DEPTH, IT_PIXEL, IT_WH, IT_SCALE, \
    IT_ROT, IT_INVERT, IT_REQUIRED, MIN_IMAGE_SIZE

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import deep_merge_dict, parse_tuple, \
    parse_number, zip_longest_fill, EnumTupleType

from Jovimetrix.sup.image import channel_solid, cv2tensor_full, image_grayscale,  \
    image_mask_add, image_rotate, image_stereogram, image_translate, pil2cv, \
    pixel_eval, tensor2cv, shape_ellipse, shape_polygon, \
    shape_quad, \
    EnumEdge, EnumImageType, \
    IT_EDGE

from Jovimetrix.sup.text import font_all, font_all_names, text_autosize, text_draw, \
    EnumAlignment, EnumJustify, EnumShapes

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"

# =============================================================================

class ConstantNode(JOVImageMultiple):
    NAME = "CONSTANT (JOV) 🟪"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Create a single RGBA block of color. Useful for masks, overlays and general filtering."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = deep_merge_dict(IT_REQUIRED, IT_WH, IT_RGBA_A)
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-constant")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        color = parse_tuple(Lexicon.RGBA_A, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)
        images = []
        params = [tuple(x) for x in zip_longest_fill(wihi, color)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (wihi, color) in enumerate(params):
            width, height = wihi
            color = pixel_eval(color, EnumImageType.BGRA)
            img = channel_solid(width, height, color, EnumImageType.BGRA)
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
            Lexicon.RGBA_A: ("VEC4", {"default": (255, 255, 255, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True, "tooltip": "Main Shape Color"}),
            Lexicon.RGB_B: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True, "tooltip": "Background Color"})
        }}
        d = deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_ROT, IT_SCALE, IT_EDGE)
        d = Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-shape-generator")
        return d

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        shape = kw.get(Lexicon.SHAPE, EnumShapes.CIRCLE)
        sides = kw.get(Lexicon.SIDES, 3)
        angle = kw.get(Lexicon.ANGLE, 0)
        edge = kw.get(Lexicon.EDGE, EnumEdge.CLIP)
        size = parse_tuple(Lexicon.SIZE, kw, EnumTupleType.FLOAT, default=(1., 1.,))
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        color = parse_tuple(Lexicon.RGBA_A, kw, default=(255, 255, 255, 255))
        bgcolor = parse_tuple(Lexicon.RGB_B, kw, default=(0, 0, 0, 255))
        params = [tuple(x) for x in zip_longest_fill(shape, sides, angle, edge, size, wihi, color, bgcolor)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (shape, sides, angle, edge, size, wihi, color, bgcolor) in enumerate(params):
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
            mask = pil2cv(mask)
            mask = image_grayscale(mask)
            img = image_mask_add(img, mask)
            img = image_rotate(img, angle, edge=edge)
            bgcolor = pixel_eval(bgcolor)
            img = cv2tensor_full(img, bgcolor)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))

class TextNode(JOVImageMultiple):
    NAME = "TEXT GENERATOR (JOV) 📝"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Use any system font with auto-fit or manual placement."
    INPUT_IS_LIST = True
    FONT_NAMES = font_all_names()
    FONTS = font_all()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
        Lexicon.STRING: ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False, "tooltip": "Your Message"}),
            Lexicon.FONT: (cls.FONT_NAMES, {"default": cls.FONT_NAMES[0]}),
            Lexicon.LETTER: ("BOOLEAN", {"default": False}),
            Lexicon.AUTOSIZE: ("BOOLEAN", {"default": False}),
            Lexicon.RGBA_A: ("VEC3", {"default": (255, 255, 255, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True, "tooltip": "Color of the letters"}),
            Lexicon.MATTE: ("VEC3", {"default": (0, 0, 0), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B], "rgb": True}),
            Lexicon.COLUMNS: ("INT", {"default": 0, "min": 0, "step": 1}),
            # if auto on, hide these...
            Lexicon.FONT_SIZE: ("INT", {"default": 100, "min": 1, "step": 1}),
            Lexicon.ALIGN: (EnumAlignment._member_names_, {"default": EnumAlignment.CENTER.name}),
            Lexicon.JUSTIFY: (EnumJustify._member_names_, {"default": EnumJustify.CENTER.name}),
            Lexicon.MARGIN: ("INT", {"default": 0, "min": -1024, "max": 1024}),
            Lexicon.SPACING: ("INT", {"default": 25, "min": -1024, "max": 1024}),
        }}
        d = deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_TRANS, IT_ROT, IT_EDGE, IT_INVERT)
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-text-generator")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        if len(full_text := kw.get(Lexicon.STRING, [""])) == 0:
            full_text = [""]
        font_idx = kw.get(Lexicon.FONT, self.FONT_NAMES[0])
        autosize = kw.get(Lexicon.AUTOSIZE, [False])
        letter = kw.get(Lexicon.LETTER, [False])
        color = parse_tuple(Lexicon.RGBA_A, kw, default=(255, 255, 255, 255))
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0), clip_min=0, clip_max=255)
        columns = kw.get(Lexicon.COLUMNS, [0])
        size = kw.get(Lexicon.FONT_SIZE, [100])
        align = kw.get(Lexicon.ALIGN, [EnumAlignment.CENTER])
        justify = kw.get(Lexicon.JUSTIFY, [EnumJustify.CENTER])
        margin = kw.get(Lexicon.MARGIN, [0])
        line_spacing = kw.get(Lexicon.SPACING, [25])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        pos = parse_tuple(Lexicon.XY, kw, EnumTupleType.FLOAT, (0, 0), -1, 1)
        angle = parse_number(Lexicon.ANGLE, kw, EnumTupleType.FLOAT, [0])
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        images = []
        params = [tuple(x) for x in zip_longest_fill(full_text, font_idx, autosize, letter, color, matte, columns, size, align, justify, margin, line_spacing, wihi, pos, angle, edge)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (full_text, font_idx, autosize, letter, color, matte, columns, size, align, justify, margin, line_spacing, wihi, pos, angle, edge) in enumerate(params):

            width, height = wihi
            font_name = self.FONTS[font_idx]
            align = EnumAlignment[align]
            justify = EnumJustify[justify]
            edge = EnumEdge[edge]
            matte = pixel_eval(matte)

            wm = width-margin*2
            hm = height-margin*2-line_spacing
            if letter:
                full_text = full_text.replace('\n', '')
                if autosize:
                    x, n = 0, 10000
                    for ch in full_text:
                        if (size := text_autosize(ch, font_name, wm, hm)[1]) > 0:
                            x = max(x, size)
                            n = min(n, size)
                    size = (x + n) * 0.25

                font = ImageFont.truetype(font_name, size)
                for ch in full_text:
                    img = text_draw(ch, font, width, height, align, justify, color=color)
                    images.append(img)

            elif autosize:
                full_text, size = text_autosize(full_text, font_name, wm, hm)[:2]
                font = ImageFont.truetype(font_name, size)
                img = text_draw(full_text, font, width, height, align, justify, margin, line_spacing, color)
                images.append(img)

        out = []
        for i in images:
            img = image_rotate(i, angle, edge=edge)
            img = image_translate(img, pos, edge=edge)
            img = cv2tensor_full(img, matte)
            out.append(img)
            pbar.update_absolute(idx)
        return list(zip(*out))

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
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL, IT_DEPTH, d)
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-stereogram")

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
            img = tensor2cv(img)
            depth = tensor2cv(depth)
            img = image_stereogram(img, depth, divisions, noise, gamma, shift)
            img = cv2tensor_full(img)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))
