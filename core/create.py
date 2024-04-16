"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import torch
import numpy as np
from PIL import ImageFont

from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOVBaseNode, JOV_WEB_RES_ROOT, WILDCARD

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_dynamic, parse_list_value, zip_longest_fill, \
    EnumConvertType

from Jovimetrix.sup.image import  cv2tensor, cv2tensor_full, \
    image_gradient, image_grayscale, image_invert, image_mask_add, image_matte, \
    image_rotate, image_stereogram, image_transform, image_translate, pil2cv, \
    pixel_eval, tensor2cv, shape_ellipse, shape_polygon, shape_quad, \
    EnumEdge, EnumImageType, MIN_IMAGE_SIZE

from Jovimetrix.sup.text import font_names, text_autosize, text_draw, \
    EnumAlignment, EnumJustify, EnumShapes

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
        pA = parse_list_value(kw.get(Lexicon.PIXEL, None), EnumConvertType.IMAGE, None)
        wihi = parse_list_value(kw.get(Lexicon.WH, None), EnumConvertType.VEC2INT, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), MIN_IMAGE_SIZE)
        matte = parse_list_value(kw.get(Lexicon.RGBA_A, None), EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        images = []
        params = zip_longest_fill(pA, wihi, matte)
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi, matte) in enumerate(params):
            width, height = wihi
            print(wihi, matte)
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
        shape = parse_list_value(kw.get(Lexicon.SHAPE, None), EnumConvertType.STRING, EnumShapes.CIRCLE.name)
        sides = parse_list_value(kw.get(Lexicon.SIDES, None), EnumConvertType.INT, 3, 3, 512)
        angle = parse_list_value(kw.get(Lexicon.ANGLE, None), EnumConvertType.FLOAT, 0)
        edge = parse_list_value(kw.get(Lexicon.EDGE, None), EnumConvertType.STRING, EnumEdge.CLIP.name)
        offset = parse_list_value(kw.get(Lexicon.XY, None), EnumConvertType.VEC2, (0, 0))
        size = parse_list_value(kw.get(Lexicon.SIZE, None), EnumConvertType.VEC2, (1, 1,), zero=0.001)
        print(kw.get(Lexicon.WH, None))
        wihi = parse_list_value(kw.get(Lexicon.WH, None), EnumConvertType.VEC2INT, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), MIN_IMAGE_SIZE)
        print(wihi)
        color = parse_list_value(kw.get(Lexicon.RGBA_A, None), EnumConvertType.VEC4INT, (255, 255, 255, 255), 0, 255)
        matte = parse_list_value(kw.get(Lexicon.MATTE, None), EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        params = zip_longest_fill(shape, sides, offset, angle, edge, size, wihi, color, matte)
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
        full_text = parse_list_value(kw.get(Lexicon.STRING, None), EnumConvertType.STRING, "")
        font_idx = parse_list_value(kw.get(Lexicon.FONT, None), EnumConvertType.STRING, self.FONT_NAMES[0])
        autosize = parse_list_value(kw.get(Lexicon.AUTOSIZE, None), EnumConvertType.BOOLEAN, False)
        letter = parse_list_value(kw.get(Lexicon.LETTER, None), EnumConvertType.BOOLEAN, False)
        color = parse_list_value(kw.get(Lexicon.RGBA_A, None), 255, (255, 255, 255, 255), EnumConvertType.VEC4INT, 0)
        matte = parse_list_value(kw.get(Lexicon.MATTE, None), 255, (0, 0, 0), EnumConvertType.VEC3INT, 0)
        columns = parse_list_value(kw.get(Lexicon.COLUMNS, None), 0, 0, EnumConvertType.INT)
        font_size = parse_list_value(kw.get(Lexicon.FONT_SIZE, None), 1, 16, EnumConvertType.INT)
        align = parse_list_value(kw.get(Lexicon.ALIGN, None), EnumConvertType.STRING, EnumAlignment.CENTER.name)
        justify = parse_list_value(kw.get(Lexicon.JUSTIFY, None), EnumConvertType.STRING, EnumJustify.CENTER.name)
        margin = parse_list_value(kw.get(Lexicon.MARGIN, None), 0, 0, EnumConvertType.INT)
        line_spacing = parse_list_value(kw.get(Lexicon.SPACING, None), 0, 25, EnumConvertType.INT)
        wihi = parse_list_value(kw.get(Lexicon.WH, None), 1, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT)
        pos = parse_list_value(kw.get(Lexicon.XY, None), 1, (0, 0), EnumConvertType.VEC2, -1)
        angle = parse_list_value(kw.get(Lexicon.ANGLE, None), EnumConvertType.INT, 0)
        edge = parse_list_value(kw.get(Lexicon.EDGE, None), EnumConvertType.STRING, EnumEdge.CLIP.name)
        invert = parse_list_value(kw.get(Lexicon.INVERT, None), EnumConvertType.BOOLEAN, False)
        images = []
        params = zip_longest_fill(full_text, font_idx, autosize, letter, color,
                                  matte, columns, font_size, align, justify, margin,
                                  line_spacing, wihi, pos, angle, edge, invert)

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
        pA = parse_list_value(kw.get(Lexicon.PIXEL, None), EnumConvertType.IMAGE, None)
        depth = parse_list_value(kw.get(Lexicon.DEPTH, None), EnumConvertType.IMAGE, None)
        divisions = parse_list_value(kw.get(Lexicon.TILE, None), 1, 8, EnumConvertType.INT)
        noise = parse_list_value(kw.get(Lexicon.NOISE, None), 1, 0.33, EnumConvertType.FLOAT, 0)
        gamma = parse_list_value(kw.get(Lexicon.GAMMA, None), 1, 0.33, EnumConvertType.FLOAT, 0)
        shift = parse_list_value(kw.get(Lexicon.SHIFT, None), 1, 1, EnumConvertType.FLOAT, -1)
        params = zip_longest_fill(pA, depth, divisions, noise, gamma, shift)
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
        pA = parse_list_value(kw.get(Lexicon.PIXEL, None), EnumConvertType.IMAGE, None)
        wihi = parse_list_value(kw.get(Lexicon.WH, None), 1, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), EnumConvertType.VEC2INT)
        colors = parse_dynamic(Lexicon.COLOR, kw)
        images = []
        params = zip_longest_fill(pA, wihi, colors)
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

class StereoscopicNode(JOVBaseNode):
    NAME = "STEREOSCOPIC (JOV) 🕶️"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {"tooltip":"Optional Image to Matte with Selected Color"}),
            Lexicon.INT: ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01, "tooltip":"Baseline"}),
            Lexicon.VALUE: ("FLOAT", {"default": 500, "min": 0, "step": 0.01, "tooltip":"Focal length"}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = parse_list_value(kw.get(Lexicon.PIXEL, None), EnumConvertType.IMAGE, None)
        baseline = parse_list_value(kw.get(Lexicon.INT, None), 1, 0.1, EnumConvertType.FLOAT)
        focal_length = parse_dynamic(Lexicon.VALUE, kw, EnumConvertType.FLOAT)
        images = []
        params = zip_longest_fill(pA, baseline, focal_length)
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi, clr) in enumerate(params):
            pA = tensor2cv(pA, EnumImageType.GRAYSCALE)

            # Convert depth image to disparity map
            disparity_map = np.divide(1.0, pA.astype(np.float32), where=pA!=0)
            # Compute disparity values based on baseline and focal length
            disparity_map *= baseline * focal_length

            images.append(cv2tensor(pA))
            pbar.update_absolute(idx)
        return list(zip(*images)) # [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]
