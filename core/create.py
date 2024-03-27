"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import torch
from PIL import ImageFont
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOV_WEB_RES_ROOT, JOVBaseNode, WILDCARD

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_dynamic, parse_parameter, zip_longest_fill, \
    EnumConvertType

from Jovimetrix.sup.image import  cv2tensor_full, \
    image_gradient, image_grayscale, image_invert, image_mask_add, image_matte, \
    image_rotate, image_stereogram, image_transform, image_translate, pil2cv, \
    pixel_eval, tensor2cv, shape_ellipse, shape_polygon, shape_quad, \
    EnumEdge, EnumImageType, MIN_IMAGE_SIZE

from Jovimetrix.sup.text import font_names, text_autosize, text_draw, \
    EnumAlignment, EnumJustify, EnumShapes

from Jovimetrix.sup.fractal import EnumNoise

# =============================================================================

JOV_CATEGORY = "CREATE"

# =============================================================================

class ConstantNode(JOVBaseNode):
    NAME = "CONSTANT (JOV) 🟪"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)

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
        pA = parse_parameter(Lexicon.PIXEL, kw, None, EnumConvertType.IMAGE)
        wihi = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT, 1)
        matte = parse_parameter(Lexicon.RGBA_A, kw, (0, 0, 0, 255), EnumConvertType.VEC4INT, 0, 255)
        images = []
        params = [tuple(x) for x in zip_longest_fill(pA, wihi, matte)]
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi, matte) in enumerate(params):
            width, height = wihi
            matte = pixel_eval(matte, EnumImageType.BGRA)
            pA = tensor2cv(pA, EnumImageType.BGRA, width, height, matte)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class ShapeNode(JOVBaseNode):
    NAME = "SHAPE GENERATOR (JOV) ✨"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)

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

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        shape = parse_parameter(Lexicon.SHAPE, kw, EnumShapes.CIRCLE.name, EnumConvertType.STRING)
        sides = parse_parameter(Lexicon.SIDES, kw, 3, EnumConvertType.INT, 3, 512)
        angle = parse_parameter(Lexicon.ANGLE, kw, 0, EnumConvertType.FLOAT)
        edge = parse_parameter(Lexicon.EDGE, kw, EnumEdge.CLIP.name, EnumConvertType.STRING)
        offset = parse_parameter(Lexicon.XY, kw, (0, 0), EnumConvertType.VEC2)
        size = parse_parameter(Lexicon.SIZE, kw, (1, 1,), EnumConvertType.VEC2, zero=0.001)
        wihi = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT, 1)
        color = parse_parameter(Lexicon.RGBA_A, kw, (255, 255, 255, 255), EnumConvertType.VEC4INT, 0, 255)
        matte = parse_parameter(Lexicon.MATTE, kw, (0, 0, 0, 255), EnumConvertType.VEC4INT, 0, 255)
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
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class TextNode(JOVBaseNode):
    NAME = "TEXT GENERATOR (JOV) 📝"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    # OUTPUT_IS_LIST = ()
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

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        full_text = parse_parameter(Lexicon.STRING, kw, "", EnumConvertType.STRING)
        font_idx = parse_parameter(Lexicon.FONT, kw, self.FONT_NAMES[0], EnumConvertType.STRING)
        autosize = parse_parameter(Lexicon.AUTOSIZE, kw, False, EnumConvertType.BOOLEAN)
        letter = parse_parameter(Lexicon.LETTER, kw, False, EnumConvertType.BOOLEAN)
        color = parse_parameter(Lexicon.RGBA_A, kw, (255, 255, 255, 255), EnumConvertType.VEC4INT, 0, 255)
        matte = parse_parameter(Lexicon.MATTE, kw, (0, 0, 0), EnumConvertType.VEC3INT, 0, 255)
        columns = parse_parameter(Lexicon.COLUMNS, kw, 0, EnumConvertType.INT, 1)
        font_size = parse_parameter(Lexicon.FONT_SIZE, kw, 16, EnumConvertType.INT, 1)
        align = parse_parameter(Lexicon.ALIGN, kw, EnumAlignment.CENTER.name, EnumConvertType.STRING)
        justify = parse_parameter(Lexicon.JUSTIFY, kw, EnumJustify.CENTER.name, EnumConvertType.STRING)
        margin = parse_parameter(Lexicon.MARGIN, kw, 0, EnumConvertType.INT, 0)
        line_spacing = parse_parameter(Lexicon.SPACING, kw, 25, EnumConvertType.INT, 0)
        wihi = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT, 1)
        pos = parse_parameter(Lexicon.XY, kw, (0, 0), EnumConvertType.VEC2, -1, 1)
        angle = parse_parameter(Lexicon.ANGLE, kw, 0, EnumConvertType.INT)
        edge = parse_parameter(Lexicon.EDGE, kw, EnumEdge.CLIP.name, EnumConvertType.STRING)
        invert = parse_parameter(Lexicon.INVERT, kw, False, EnumConvertType.BOOLEAN)
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
            wm = width-margin * 2
            hm = height-margin * 2 - line_spacing
            if letter:
                full_text = full_text.replace('\n', '')
                if autosize:
                    w, h = text_autosize(full_text, font_name, wm, hm)[2:]
                    w /= len(full_text) * 1.25 # kerning?
                    font_size = (w + h) * 0.5
                font_size *= 10
                font = ImageFont.truetype(font_name, int(font_size))
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
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class StereogramNode(JOVBaseNode):
    NAME = "STEREOGRAM (JOV) 📻"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    # OUTPUT_IS_LIST = ()

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

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = parse_parameter(Lexicon.PIXEL, kw, None, EnumConvertType.IMAGE)
        depth = parse_parameter(Lexicon.DEPTH, kw, None, EnumConvertType.IMAGE)
        divisions = parse_parameter(Lexicon.TILE, kw, 8, EnumConvertType.INT, 1)
        noise = parse_parameter(Lexicon.NOISE, kw, 0.33, EnumConvertType.FLOAT, 0, 1)
        gamma = parse_parameter(Lexicon.GAMMA, kw, 0.33, EnumConvertType.FLOAT, 0, 1)
        shift = parse_parameter(Lexicon.SHIFT, kw, 1, EnumConvertType.FLOAT, -1, 1)
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
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class GradientNode(JOVBaseNode):
    NAME = "GRADIENT (JOV) 🍧"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)

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
        pA = parse_parameter(Lexicon.PIXEL, kw, None, EnumConvertType.IMAGE)
        wihi = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT, 1)
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
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]
