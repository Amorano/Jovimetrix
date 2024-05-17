"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from typing import Tuple

import torch
import numpy as np
from PIL import ImageFont
from skimage.filters import gaussian

from comfy.utils import ProgressBar

from Jovimetrix import JOVBaseNode, JOV_WEB_RES_ROOT, WILDCARD

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_dynamic, parse_param, zip_longest_fill, \
    EnumConvertType

from Jovimetrix.sup.image import  channel_solid, cv2tensor, cv2tensor_full, \
    image_grayscale, image_invert, image_mask_add, \
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

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)], MIN_IMAGE_SIZE)
        matte = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        images = []
        params = list(zip_longest_fill(pA, wihi, matte))
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi, matte) in enumerate(params):
            width, height = wihi
            matte = pixel_eval(matte, EnumImageType.BGRA)
            channel_solid(width, height, matte, EnumImageType.BGRA)
            pA = tensor2cv(pA) if pA is not None else channel_solid(width, height, matte, EnumImageType.BGRA)
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
            Lexicon.BLUR: ("FLOAT", {"default": 0, "min": 0, "step": 0.01, "precision": 4,
                                    "round": 0.00001, "tooltip": "Edge blur amount (Gaussian blur)"}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = parse_param(kw, Lexicon.SHAPE, EnumConvertType.STRING, EnumShapes.CIRCLE.name)
        sides = parse_param(kw, Lexicon.SIDES, EnumConvertType.INT, 3, 3, 512)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        edge = parse_param(kw, Lexicon.EDGE, EnumConvertType.STRING, EnumEdge.CLIP.name)
        offset = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, (0, 0))
        size = parse_param(kw, Lexicon.SIZE, EnumConvertType.VEC2, (1, 1,), zero=0.001)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), MIN_IMAGE_SIZE)
        color = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, (255, 255, 255, 255), 0, 255)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        blur = parse_param(kw, Lexicon.BLUR, EnumConvertType.FLOAT, 0)
        params = list(zip_longest_fill(shape, sides, offset, angle, edge, size, wihi, color, matte, blur))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (shape, sides, offset, angle, edge, size, wihi, color, matte, blur) in enumerate(params):
            width, height = wihi
            sizeX, sizeY = size
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
            if blur > 0:
                pA = (gaussian(pA, sigma=blur, channel_axis=2) * 255).astype(np.uint8)
                mask = (gaussian(mask, sigma=blur, channel_axis=2) * 255).astype(np.uint8)
            # logger.debug(pA.shape, mask.shape)
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

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        full_text = parse_param(kw, Lexicon.STRING, EnumConvertType.STRING, "")
        font_idx = parse_param(kw, Lexicon.FONT, EnumConvertType.STRING, self.FONT_NAMES[0])
        autosize = parse_param(kw, Lexicon.AUTOSIZE, EnumConvertType.BOOLEAN, False)
        letter = parse_param(kw, Lexicon.LETTER, EnumConvertType.BOOLEAN, False)
        color = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(255, 255, 255, 255)], 0, 255)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC3INT, 0, 255)
        columns = parse_param(kw, Lexicon.COLUMNS, EnumConvertType.INT, 0)
        font_size = parse_param(kw, Lexicon.FONT_SIZE, EnumConvertType.INT, 1)
        align = parse_param(kw, Lexicon.ALIGN, EnumConvertType.STRING, EnumAlignment.CENTER.name)
        justify = parse_param(kw, Lexicon.JUSTIFY, EnumConvertType.STRING, EnumJustify.CENTER.name)
        margin = parse_param(kw, Lexicon.MARGIN, EnumConvertType.INT, 0)
        line_spacing = parse_param(kw, Lexicon.SPACING, EnumConvertType.INT, 25)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)], MIN_IMAGE_SIZE)
        pos = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, [(0, 0)], 1,  -1)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.INT, 0)
        edge = parse_param(kw, Lexicon.EDGE, EnumConvertType.STRING, EnumEdge.CLIP.name)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        images = []
        params = list(zip_longest_fill(full_text, font_idx, autosize, letter, color,
                                  matte, columns, font_size, align, justify, margin,
                                  line_spacing, wihi, pos, angle, edge, invert))

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

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        depth = parse_param(kw, Lexicon.DEPTH, EnumConvertType.IMAGE, None)
        divisions = parse_param(kw, Lexicon.TILE, EnumConvertType.INT, 1, 1, 8)
        noise = parse_param(kw, Lexicon.NOISE, EnumConvertType.FLOAT, 1, 0)
        gamma = parse_param(kw, Lexicon.GAMMA, EnumConvertType.FLOAT, 1, 0)
        shift = parse_param(kw, Lexicon.SHIFT, EnumConvertType.FLOAT, 0, 1, -1)
        params = list(zip_longest_fill(pA, depth, divisions, noise, gamma, shift))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, depth, divisions, noise, gamma, shift) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            h, w = pA.shape[:2]
            depth = tensor2cv(depth) if depth is not None else channel_solid(w, h, chan=EnumImageType.BGRA)
            images.append(cv2tensor_full(pA))
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

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        baseline = parse_param(kw, Lexicon.INT, 1, 0.1, EnumConvertType.FLOAT)
        focal_length = parse_dynamic(Lexicon.VALUE, kw, EnumConvertType.FLOAT)
        images = []
        params = list(zip_longest_fill(pA, baseline, focal_length))
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi, clr) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.GRAYSCALE)
            # Convert depth image to disparity map
            disparity_map = np.divide(1.0, pA.astype(np.float32), where=pA!=0)
            # Compute disparity values based on baseline and focal length
            disparity_map *= baseline * focal_length

            images.append(cv2tensor(pA))
            pbar.update_absolute(idx)
        return list(zip(*images)) # [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]
