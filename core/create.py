"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from enum import Enum
import numpy as np

import torch
from PIL import ImageFont
from vnoise import Noise
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import WILDCARD, JOVImageSimple, JOVImageMultiple, \
    JOV_HELP_URL, MIN_IMAGE_SIZE

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_dynamic, parse_tuple, zip_longest_fill, \
    EnumTupleType

from Jovimetrix.sup.image import batch_extract, channel_solid, cv2tensor_full, \
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
    DESCRIPTION = "Create a single RGBA block of color. Useful for masks, overlays and general filtering."

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
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-constant")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = batch_extract(kw.get(Lexicon.PIXEL, None))
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        matte = parse_tuple(Lexicon.RGBA_A, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)
        images = []
        params = [tuple(x) for x in zip_longest_fill(pA, wihi, matte)]
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi, matte) in enumerate(params):
            width, height = wihi
            matte = pixel_eval(matte, EnumImageType.BGRA)
            if pA is None:
                pA = channel_solid(width, height, matte, EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return list(zip(*images))

class ShapeNode(JOVImageMultiple):
    NAME = "SHAPE GENERATOR (JOV) ✨"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Generate polyhedra for masking or texture work."

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
        d = Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-shape-generator")
        return d

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        shape = kw.get(Lexicon.SHAPE, [EnumShapes.CIRCLE])
        sides = kw.get(Lexicon.SIDES, [3])
        angle = kw.get(Lexicon.ANGLE, [0])
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        offset = parse_tuple(Lexicon.XY, kw, typ=EnumTupleType.FLOAT, default=(0., 0.,))
        size = parse_tuple(Lexicon.SIZE, kw, EnumTupleType.FLOAT, default=(1., 1.,))
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        color = parse_tuple(Lexicon.RGBA_A, kw, default=(255, 255, 255, 255))
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0, 255))
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
            pA = image_transform(pA, offset, angle, size, edge=edge)
            matte = pixel_eval(matte, EnumImageType.BGRA)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return list(zip(*images))

class TextNode(JOVImageMultiple):
    NAME = "TEXT GENERATOR (JOV) 📝"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Use any system font with auto-fit or manual placement."
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
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-text-generator")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        if len(full_text := kw.get(Lexicon.STRING, [""])) == 0:
            full_text = [""]
        font_idx = kw.get(Lexicon.FONT, [self.FONT_NAMES[0]])
        autosize = kw.get(Lexicon.AUTOSIZE, [False])
        letter = kw.get(Lexicon.LETTER, [False])
        color = parse_tuple(Lexicon.RGBA_A, kw, default=(255, 255, 255, 255))
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0), clip_min=0, clip_max=255)
        columns = kw.get(Lexicon.COLUMNS, [0])
        font_size = kw.get(Lexicon.FONT_SIZE, [16])
        align = kw.get(Lexicon.ALIGN, [EnumAlignment.CENTER])
        justify = kw.get(Lexicon.JUSTIFY, [EnumJustify.CENTER])
        margin = kw.get(Lexicon.MARGIN, [0])
        line_spacing = kw.get(Lexicon.SPACING, [25])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))
        pos = parse_tuple(Lexicon.XY, kw, EnumTupleType.FLOAT, (0, 0), -1, 1)
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
    DESCRIPTION = "Make a magic eye stereograms."
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
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-stereogram")

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
            if pA is None:
                pA = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, chan=EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)

            depth = tensor2cv(depth)
            pA = image_stereogram(pA, depth, divisions, noise, gamma, shift)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return list(zip(*images))

class GradientNode(JOVImageMultiple):
    NAME = "GRADIENT (JOV) 🍧"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Make a gradient mapped to a linear or polar coordinate system."

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
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-constant")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = batch_extract(kw.get(Lexicon.PIXEL, None))
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
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

class NoiseNode(JOVImageMultiple):
    NAME = "NOISE (JOV) 🏞"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Blocks of noise"
    RETURN_TYPES = ("FLOAT", "IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = (Lexicon.FLOAT, Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK,)
    OUTPUT_IS_LIST = (True, True, True, True, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.NOISE: (EnumNoise._member_names_, {"default": EnumNoise.PERLIN_2D.name}),
            Lexicon.SEED: ("INT", {"default": 0, "step": 1}),
            Lexicon.X: ("FLOAT", {"default": MIN_IMAGE_SIZE, "step": 1,
                                  "label": Lexicon.X, "min": 2,
                                  "tooltip": "Width of the Noise Output"}),
            Lexicon.XY: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1,
                                  "label": [Lexicon.X, Lexicon.Y],
                                  "tooltip": "Width and Height of the Noise Output"}),
            #Lexicon.XYZ: ("VEC3", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), "step": 1,
            #                      "label": [Lexicon.X, Lexicon.Y, Lexicon.Z],
            #                      "tooltip": "Width, Height and Depth of the Noise Output"}),
            #Lexicon.XYZW: ("VEC4", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4, 1), "step": 1,
            #                      "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
            #                      "tooltip": "Width, Height, Depth and Time of the Noise Output"}),
            Lexicon.INDEX: ("INT", {"default": 1, "step": 1, "min": 1, "max": 2, "tooltip":"which index in the noise block to use for the float output"}),
            Lexicon.OCTAVES: ("INT", {"default": 2, "step": 1, "min": 1, "max": 32, "tooltip":"number of passes"}),
            Lexicon.PERSISTENCE: ("FLOAT", {"default": 2, "min": 0, "step": 0.01, "tooltip":"relative amplitude of each octave to its parent"}),
            Lexicon.LACUNARITY: ("FLOAT", {"default": 4, "min": 0, "step": 0.01, "tooltip":"frequency of each successive octave relative"}),
            Lexicon.OFFSET: ("INT", {"default": 0, "step": 1}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE*2, MIN_IMAGE_SIZE*2), "step": 1,
                                  "label": [Lexicon.W, Lexicon.H],
                                  "tooltip": "Where the noise will repeat"}),
            Lexicon.VALUE: ("FLOAT", {"default": 255, "step": 0.1, "tooltip":"scale the noise"}),
            Lexicon.ROUND: ("INT", {"default": 0, "step": 1, "min": 0, "max": 16}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-noise")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__seed = 0
        self.__noise = None
        self.__ntype = None
        self.__empty = (torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=torch.uint8, device="cpu"),
                        torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu"),
                        torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 1), dtype=torch.uint8, device="cpu"),)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        ntype = kw.get(Lexicon.NOISE, [EnumNoise.PERLIN_2D_RGB])
        seed = kw.get(Lexicon.SEED, [0])
        octaves = kw.get(Lexicon.OCTAVES, [2])
        persistence = kw.get(Lexicon.PERSISTENCE, [2])
        lacunarity = kw.get(Lexicon.LACUNARITY, [4])
        float_index = kw.get(Lexicon.INDEX, [0])
        x = kw.get(Lexicon.X, [512])
        xy = parse_tuple(Lexicon.XY, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=2)
        #xyz = parse_tuple(Lexicon.XYZ, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), clip_min=1)
        #xyzw = parse_tuple(Lexicon.XYZW, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4, 1), clip_min=1)
        offset = kw.get(Lexicon.OFFSET, [0])
        repeat = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE*2, MIN_IMAGE_SIZE*2,), clip_min=1)
        scalar = kw.get(Lexicon.VALUE, [255])
        rounding = kw.get(Lexicon.ROUND, [0])
        images = []
        params = [tuple(x) for x in zip_longest_fill(ntype, seed, octaves, persistence, lacunarity, float_index, x, xy, offset, repeat, scalar, rounding)]
        pbar = ProgressBar(len(params))
        for idx, (ntype, seed, octaves, persistence, lacunarity, float_index, x, xy, offset, repeat, scalar, rounding) in enumerate(params):
            ntype = EnumNoise[ntype]
            if self.__ntype != ntype or self.__noise is None or self.__seed != seed:
                self.__noise = Noise(seed)
                #if ntype in [EnumNoise.PERLIN_1D, EnumNoise.PERLIN_2D, EnumNoise.PERLIN_2D_RGB, EnumNoise.#PERLIN_2D_RGBA]:
                    #self.__noise = Noise(seed)
                #else:
                    #self.__noise = SNoise(seed)
                block = None
                self.__ntype = ntype

            floats = []
            block = []
            repeat_x, repeat_y = repeat
            if ntype == EnumNoise.PERLIN_1D:
                floats = block = self.__noise.noise1(np.linspace(0, 1, x), octaves, persistence, lacunarity, repeat_x, offset)
                image = self.__empty
            else:
                x, y = xy
                float_index -= 1
                if ntype == EnumNoise.PERLIN_2D:
                    block = self.__noise.noise2(np.linspace(0, 1, x), np.linspace(0, 1, y), octaves, persistence, lacunarity, repeat_x, repeat_y, offset)
                elif ntype == EnumNoise.PERLIN_2D_RGB:
                    block = self.__noise.noise3(np.linspace(0, 1, x), np.linspace(0, 1, y), np.linspace(0, 1, 3), octaves, persistence, lacunarity, repeat_x, repeat_y, repeat_x, offset)
                elif ntype == EnumNoise.PERLIN_2D_RGBA:
                    block = self.__noise.noise3(np.linspace(0, 1, x), np.linspace(0, 1, y), np.linspace(0, 1, 4), octaves, persistence, lacunarity, repeat_x, repeat_y, repeat_x, offset)

                image = np.array(block * scalar, dtype=np.uint8)
                image = cv2tensor_full(image)
                floats = block[float_index]

            if rounding > 0:
                rounding = max(0, min(16, rounding))
                scalar = round(scalar, rounding)
            else:
                scalar = int(scalar)
            floats *= scalar
            if rounding == 0:
                floats = floats.astype(np.uint8)
            floats = [floats]
            floats.extend(image)
            images.append(floats)
            pbar.update_absolute(idx)
        return list(zip(*images))
