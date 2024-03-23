"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import numpy as np

import torch
from PIL import ImageFont
from vnoise import Noise
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import load_help, JOVImageSimple, JOVImageMultiple, \
    MIN_IMAGE_SIZE, WILDCARD

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_dynamic, parse_tuple, zip_longest_fill, \
    EnumTupleType

from Jovimetrix.sup.image import batch_extract, cv2tensor_full, \
    image_gradient, image_grayscale, image_invert, image_mask_add, image_matte, \
    image_rotate, image_stereogram, image_transform, image_translate, pil2cv, \
    pixel_eval, tensor2cv, shape_ellipse, shape_polygon, shape_quad, \
    EnumEdge, EnumImageType

from Jovimetrix.sup.text import font_names, text_autosize, text_draw, \
    EnumAlignment, EnumJustify, EnumShapes

from Jovimetrix.sup.fractal import EnumNoise

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"

# =============================================================================

class ConstantNode(JOVImageMultiple):
    NAME = "CONSTANT (JOV) 🟪"
    CATEGORY = JOV_CATEGORY
    HELP_URL = "CREATE#-constant"
    DESC = "Create a single RGBA block of color. Useful for masks, overlays and general filtering."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {"tooltip":"Optional Image to Matte with Selected Color"}),
            Lexicon.RGBA_A: ("VEC4", {"default": (0, 0, 0, 255), "step": 1,
                                      "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A],
                                      "rgb": True, "tooltip": "Constant Color to Output"}),
            Lexicon.WH: ("VEC2", {"default": (512, 512), "step": 1,
                                  "label": [Lexicon.W, Lexicon.H],
                                  "tooltip": "Desired Width and Height of the Color Output"})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = batch_extract(kw.get(Lexicon.PIXEL, None))
        wihi = parse_tuple(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        matte = parse_tuple(Lexicon.RGBA_A, kw, (0, 0, 0, 255), clip_min=0, clip_max=255)
        images = []
        params = [tuple(x) for x in zip_longest_fill(pA, wihi, matte)]
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi, matte) in enumerate(params):
            width, height = wihi
            matte = pixel_eval(matte, EnumImageType.BGRA)
            pA = tensor2cv(pA, EnumImageType.BGRA, width, height, matte)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return list(zip(*images))

class ShapeNode(JOVImageMultiple):
    NAME = "SHAPE GENERATOR (JOV) ✨"
    CATEGORY = JOV_CATEGORY
    HELP_URL = "CREATE#-shape-generator"
    DESC = "Generate polyhedra for masking or texture work."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.SHAPE: (EnumShapes._member_names_, {"default": EnumShapes.CIRCLE.name}),
            Lexicon.SIDES: ("INT", {"default": 3, "min": 3, "max": 100, "step": 1}),
            Lexicon.RGBA_A: ("VEC4", {"default": (255, 255, 255, 255), "step": 1,
                                      "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A],
                                      "rgb": True, "tooltip": "Main Shape Color"}),
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1,
                                     "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A],
                                     "rgb": True, "tooltip": "Background Color"}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE),
                                  "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.XY: ("VEC2", {"default": (0, 0,), "step": 0.01, "precision": 4,
                                   "round": 0.00001, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "min": -180, "max": 180,
                                      "step": 0.01, "precision": 4, "round": 0.00001}),
            Lexicon.SIZE: ("VEC2", {"default": (1., 1.), "step": 0.01, "precision": 4,
                                    "round": 0.00001, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)
        d = Lexicon._parse(d, "/CREATE#-shape-generator")
        return d

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        shape = kw.get(Lexicon.SHAPE, [EnumShapes.CIRCLE])
        sides = kw.get(Lexicon.SIDES, [3])
        angle = kw.get(Lexicon.ANGLE, [0])
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        offset = parse_tuple(Lexicon.XY, kw, (0., 0.,), EnumTupleType.FLOAT, )
        size = parse_tuple(Lexicon.SIZE, kw, (1., 1.,), EnumTupleType.FLOAT, )
        wihi = parse_tuple(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        color = parse_tuple(Lexicon.RGBA_A, kw, (255, 255, 255, 255))
        matte = parse_tuple(Lexicon.MATTE, kw, (0, 0, 0, 255))
        params = [tuple(x) for x in zip_longest_fill(shape, sides, offset, angle, edge,
                                                     size, wihi, color, matte)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (shape, sides, offset, angle, edge, size, wihi, color, matte) in enumerate(params):
            width, height = wihi
            sizeX, sizeY = size
            sides = int(sides)
            edge = EnumEdge[edge]
            shape = EnumShapes[shape]
            match shape:
                case EnumShapes.SQUARE:
                    pA = shape_quad(width, height, sizeX, sizeX, fill=color, back=matte)
                    mask = shape_quad(width, height, sizeX, sizeX, fill=color[3])

                case EnumShapes.ELLIPSE:
                    pA = shape_ellipse(width, height, sizeX, sizeY, fill=color, back=matte)
                    mask = shape_ellipse(width, height, sizeX, sizeY, fill=color[3])

                case EnumShapes.RECTANGLE:
                    pA = shape_quad(width, height, sizeX, sizeY, fill=color, back=matte)
                    mask = shape_quad(width, height, sizeX, sizeY, fill=color[3])

                case EnumShapes.POLYGON:
                    pA = shape_polygon(width, height, sizeX, sides, fill=color, back=matte)
                    mask = shape_polygon(width, height, sizeX, sides, fill=color[3])

                case EnumShapes.CIRCLE:
                    pA = shape_ellipse(width, height, sizeX, sizeX, fill=color, back=matte)
                    mask = shape_ellipse(width, height, sizeX, sizeX, fill=color[3])

            pA = pil2cv(pA)
            mask = pil2cv(mask)
            mask = image_grayscale(mask)
            pA = image_mask_add(pA, mask)
            pA = image_transform(pA, offset, angle, (1,1), edge=edge)
            matte = pixel_eval(matte, EnumImageType.BGRA)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return list(zip(*images))

class TextNode(JOVImageMultiple):
    NAME = "TEXT GENERATOR (JOV) 📝"
    CATEGORY = JOV_CATEGORY
    HELP_URL = "CREATE#-text-generator"
    DESC = "Use any system font with auto-fit or manual placement."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    FONTS = font_names()
    FONT_NAMES = sorted(FONTS.keys())

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.STRING: ("STRING", {"default": "", "multiline": True,
                                        "dynamicPrompts": False,
                                        "tooltip": "Your Message"}),
            Lexicon.FONT: (cls.FONT_NAMES, {"default": cls.FONT_NAMES[0]}),
            Lexicon.LETTER: ("BOOLEAN", {"default": False}),
            Lexicon.AUTOSIZE: ("BOOLEAN", {"default": False}),
            Lexicon.RGBA_A: ("VEC3", {"default": (255, 255, 255, 255), "step": 1,
                                      "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A],
                                      "rgb": True, "tooltip": "Color of the letters"}),
            Lexicon.MATTE: ("VEC3", {"default": (0, 0, 0), "step": 1,
                                     "label": [Lexicon.R, Lexicon.G, Lexicon.B], "rgb": True}),
            Lexicon.COLUMNS: ("INT", {"default": 0, "min": 0, "step": 1}),
            # if auto on, hide these...
            Lexicon.FONT_SIZE: ("INT", {"default": 16, "min": 1, "step": 1}),
            Lexicon.ALIGN: (EnumAlignment._member_names_, {"default": EnumAlignment.CENTER.name}),
            Lexicon.JUSTIFY: (EnumJustify._member_names_, {"default": EnumJustify.CENTER.name}),
            Lexicon.MARGIN: ("INT", {"default": 0, "min": -1024, "max": 1024}),
            Lexicon.SPACING: ("INT", {"default": 25, "min": -1024, "max": 1024}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE),
                                  "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.XY: ("VEC2", {"default": (0, 0,), "step": 0.01, "precision": 4,
                                  "round": 0.00001, "label": [Lexicon.X, Lexicon.Y],
                                  "tooltip":"Offset the position"}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "min": -180, "max": 180,
                                      "step": 0.01, "precision": 4, "round": 0.00001}),
            Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
            Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"})
        }}
        return Lexicon._parse(d, cls.HELP_URL)
        return Lexicon._parse(d, "/CREATE#-text-generator")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        if len(full_text := kw.get(Lexicon.STRING, [""])) == 0:
            full_text = [""]
        font_idx = kw.get(Lexicon.FONT, [self.FONT_NAMES[0]])
        autosize = kw.get(Lexicon.AUTOSIZE, [False])
        letter = kw.get(Lexicon.LETTER, [False])
        color = parse_tuple(Lexicon.RGBA_A, kw, (255, 255, 255, 255))
        matte = parse_tuple(Lexicon.MATTE, kw, (0, 0, 0), clip_min=0, clip_max=255)
        columns = kw.get(Lexicon.COLUMNS, [0])
        font_size = kw.get(Lexicon.FONT_SIZE, [16])
        align = kw.get(Lexicon.ALIGN, [EnumAlignment.CENTER])
        justify = kw.get(Lexicon.JUSTIFY, [EnumJustify.CENTER])
        margin = kw.get(Lexicon.MARGIN, [0])
        line_spacing = kw.get(Lexicon.SPACING, [25])
        wihi = parse_tuple(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        pos = parse_tuple(Lexicon.XY, kw, (0, 0), EnumTupleType.FLOAT,  -1, 1)
        angle = kw.get(Lexicon.ANGLE, [0])
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        invert = kw.get(Lexicon.INVERT, [False])
        images = []
        params = [tuple(x) for x in zip_longest_fill(full_text, font_idx, autosize,
                                                     letter, color, matte, columns,
                                                     font_size, align, justify,
                                                     margin, line_spacing, wihi,
                                                     pos, angle, edge, invert)]

        pbar = ProgressBar(len(params))
        for idx, (full_text, font_idx, autosize, letter, color, matte, columns,
                  font_size, align, justify, margin, line_spacing, wihi, pos,
                  angle, edge, invert) in enumerate(params):

            width, height = wihi
            font_name = self.FONTS[font_idx]
            align = EnumAlignment[align]
            justify = EnumJustify[justify]
            edge = EnumEdge[edge]
            matte = pixel_eval(matte)
            full_text = str(full_text)
            # color = pixel_eval(color, EnumImageType.BGRA)
            wm = width-margin * 2
            hm = height-margin * 2 - line_spacing
            if letter:
                full_text = full_text.replace('\n', '')
                if autosize:
                    w, h = text_autosize(full_text, font_name, wm, hm)[2:]
                    w /= len(full_text) * 1.25 # kerning?
                    font_size = (w + h) * 0.5
                font_size *= 10
                font = ImageFont.truetype(font_name, font_size)
                for ch in full_text:
                    img = text_draw(ch, font, width, height, align, justify, color=color)
                    img = image_rotate(img, angle, edge=edge)
                    img = image_translate(img, pos, edge=edge)
                    if invert:
                        img = image_invert(img, 1)
                    images.append(cv2tensor_full(img, matte))
            else:
                if autosize:
                    full_text, font_size = text_autosize(full_text, font_name, wm, hm, columns)[:2]
                font = ImageFont.truetype(font_name, font_size)
                img = text_draw(full_text, font, width, height, align, justify,
                                margin, line_spacing, color)
                img = image_rotate(img, angle, edge=edge)
                img = image_translate(img, pos, edge=edge)
                if invert:
                    img = image_invert(img, 1)
                images.append(cv2tensor_full(img, matte))
            pbar.update_absolute(idx)
        return list(zip(*images))

class StereogramNode(JOVImageSimple):
    NAME = "STEREOGRAM (JOV) 📻"
    CATEGORY = JOV_CATEGORY
    HELP_URL = "CREATE#-stereogram"
    DESC = "Make a magic eye stereograms."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.DEPTH: (WILDCARD, {}),
            Lexicon.TILE: ("INT", {"default": 8, "min": 1}),
            Lexicon.NOISE: ("FLOAT", {"default": 0.33, "min": 0, "max": 1, "step": 0.01}),
            Lexicon.GAMMA: ("FLOAT", {"default": 0.33, "min": 0, "max": 1, "step": 0.01}),
            Lexicon.SHIFT: ("FLOAT", {"default": 1., "min": -1, "max": 1, "step": 0.01}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)
        return Lexicon._parse(d, "/CREATE#-stereogram")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = batch_extract(kw.get(Lexicon.PIXEL, None))
        depth = batch_extract(kw.get(Lexicon.DEPTH, None))
        divisions = kw.get(Lexicon.TILE, [8])
        noise = kw.get(Lexicon.NOISE, [0.33])
        gamma = kw.get(Lexicon.GAMMA, [0.33])
        shift = kw.get(Lexicon.SHIFT, [1])
        params = [tuple(x) for x in zip_longest_fill(pA, depth, divisions, noise,
                                                     gamma, shift)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, depth, divisions, noise, gamma, shift) in enumerate(params):
            pA = tensor2cv(pA)
            depth = tensor2cv(depth)
            pA = image_stereogram(pA, depth, divisions, noise, gamma, shift)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return list(zip(*images))

class GradientNode(JOVImageMultiple):
    NAME = "GRADIENT (JOV) 🍧"
    CATEGORY = JOV_CATEGORY
    HELP_URL = "CREATE#-gradient"
    DESC = "Make a gradient mapped to a linear or polar coordinate system."
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {"tooltip":"Optional Image to Matte with Selected Color"}),
            Lexicon.WH: ("VEC2", {"default": (512, 512), "step": 1,
                                  "label": [Lexicon.W, Lexicon.H],
                                  "tooltip": "Desired Width and Height of the Color Output"})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = batch_extract(kw.get(Lexicon.PIXEL, None))
        wihi = parse_tuple(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        colors = parse_dynamic(Lexicon.COLOR, kw)
        images = []
        params = [tuple(x) for x in zip_longest_fill(pA, wihi, colors)]
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi, clr) in enumerate(params):
            # colors = [(0,0,0,255) if c is None else pixel_eval(c, EnumImageType.BGRA) for c in clr]
            width, height = wihi
            image = image_gradient(width, height, clr)
            if pA is not None:
                pA = tensor2cv(pA)
                pA = image_matte(image, imageB=pA)
            images.append(cv2tensor_full(image))
            pbar.update_absolute(idx)
        return list(zip(*images))
