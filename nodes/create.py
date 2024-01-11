"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import json
from enum import Enum

import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager
from loguru import logger

import comfy
from server import PromptServer

from Jovimetrix import ComfyAPIMessage, JOVBaseNode, JOVImageBaseNode, \
    ROOT, IT_PIXELS, IT_RGBA, IT_WH, IT_SCALE, IT_ROT, IT_INVERT, \
    IT_REQUIRED, MIN_IMAGE_SIZE

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import deep_merge_dict, parse_tuple, parse_number, \
    EnumTupleType

from Jovimetrix.sup.image import b64_2_tensor, channel_add, pil2tensor, pil2cv, \
    cv2tensor, cv2mask, IT_WHMODE

from Jovimetrix.sup.comp import shape_ellipse, shape_polygon, shape_quad, \
    light_invert, EnumScaleMode

from Jovimetrix.sup.shader import GLSL

JOV_CONFIG_GLSL = ROOT / 'glsl'

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

# =============================================================================

class ConstantNode(JOVImageBaseNode):
    NAME = "CONSTANT (JOV) 🟪"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_WH, IT_RGBA)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        color = parse_tuple(Lexicon.RGBA, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)[0]
        image = Image.new("RGB", (width, height), color)
        return (pil2tensor(image), pil2tensor(image.convert("L")),)

class ShapeNode(JOVImageBaseNode):
    NAME = "SHAPE GENERATOR (JOV) ✨"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

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
        return deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_ROT, IT_SCALE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        shape = kw.get(Lexicon.SHAPE, EnumShapes.CIRCLE)
        shape = EnumShapes[shape]
        sides = kw.get(Lexicon.SIDES, 3)
        angle = kw.get(Lexicon.ANGLE, 0)
        sizeX, sizeY = parse_tuple(Lexicon.SIZE, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))[0]
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))[0]
        color = parse_tuple(Lexicon.RGB, kw, default=(255, 255, 255))[0]
        bgcolor = parse_tuple(Lexicon.RGB_B, kw, default=(0, 0, 0))[0]
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1])[0]
        img = None
        mask = None
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

        img = img.rotate(-angle)
        mask = mask.rotate(-angle)

        img = pil2cv(img)
        mask = pil2cv(mask)
        if i != 0:
            img = light_invert(img, i)
            mask = light_invert(mask, i)

        img = channel_add(img, 0)
        img[:, :, 3] = mask[:,:,0]
        return (cv2tensor(img), cv2tensor(mask), )

class TextNode(JOVImageBaseNode):
    NAME = "TEXT GENERATOR (JOV) 📝"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.STRING: ("STRING", {"default": "", "multiline": True}),
                Lexicon.FONT: (FONT_NAMES, {"default": FONT_NAMES[0]}),
                Lexicon.FONT_SIZE: ("FLOAT", {"default": 10, "min": 1, "step": 0.01}),
                Lexicon.RGB: ("VEC3", {"default": (255, 255, 255), "min": 0, "max": 255, "step": 1, "label":
                                       [Lexicon.R, Lexicon.G, Lexicon.B]}),
                Lexicon.RGB_B: ("VEC3", {"default": (0, 0, 0), "min": 0, "max": 255, "step": 1, "label":
                                       [Lexicon.R, Lexicon.G, Lexicon.B]})
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WHMODE, IT_ROT, IT_SCALE, IT_INVERT)

    @staticmethod
    def render_text(text, font_path, font_size, color, bgcolor, width,
                    height, autofit=False) -> Image:

        font = ImageFont.truetype(font_path, font_size)
        img = Image.new("RGB", (width, height), bgcolor)
        draw = ImageDraw.Draw(img)
        draw.multiline_text((0, 0), text, font=font, fill=color)
        return img

        if autofit:
            img = Image.new("RGB", (1, 1), bgcolor)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, font_size)

            width, height = draw.multiline_text(text, font)

        img = Image.new("RGB", (width, height), bgcolor)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font, fill=color)
        return img

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        text = kw.get(Lexicon.STRING, "")
        font = FONTS[kw[Lexicon.FONT]]
        font_size = kw[Lexicon.FONT_SIZE]
        mode = EnumScaleMode[kw[Lexicon.MODE]]
        rot = parse_number(Lexicon.ANGLE, kw, EnumTupleType.FLOAT, [1])[0]
        sizeX, sizeY = parse_tuple(Lexicon.SIZE, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))[0]
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,))[0]
        color = parse_tuple(Lexicon.RGB, kw, default=(255, 255, 255))[0]
        bgcolor = parse_tuple(Lexicon.RGB_B, kw, default=(0, 0, 0))[0]
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1])[0]

        img = TextNode.render_text(text, font, font_size, color, bgcolor, width, height, autofit=mode == EnumScaleMode.NONE)
        img = pil2cv(img)
        if i != 0:
            img = light_invert(img, i)
        return (cv2tensor(img), cv2mask(img),)

class GLSLNode(JOVBaseNode):
    NAME = "GLSL (JOV) 🍩"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        # PRESETS = list_shaders(JOV_CONFIG_GLSL)
        # PRESET = next(iter(PRESETS)) if len(PRESETS) else ""
        d = {"optional": {
                Lexicon.TIME: ("FLOAT", {"default": 0, "step": 0.0001, "min": 0, "precision": 6}),
                Lexicon.FPS: ("INT", {"default": 0, "step": 1, "min": 0, "max": 120}),
                Lexicon.BATCH: ("INT", {"default": 1, "step": 1, "min": 1, "max": 36000}),
                Lexicon.RESET: ("BOOLEAN", {"default": False}),
                Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), "step": 1, "min": 1}),
                # Lexicon.PRESET: ([""], {"default": ""}),
                Lexicon.FRAGMENT: ("STRING", {"multiline": True, "default": "void main() {\n\tfragColor = vec4(iUV, 0, 1.0);\n}"}),
                Lexicon.USER1: ("FLOAT", {"default": 0, "step": 0.0001, "precision": 6}),
                Lexicon.USER2: ("FLOAT", {"default": 0, "step": 0.0001, "precision": 6}),
            },
            "hidden": {
                "id": "UNIQUE_ID"
            }}

        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        self.__last = torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8)

    def run(self, id, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        # ask for this "id"'s image
        PromptServer.instance.send_sync("jovi-glsl-image", {"id": id})
        self.__last = []
        batch = kw.get(Lexicon.BATCH, 1)
        imgs = ComfyAPIMessage.poll(id, timeout=(batch / 10) + 3)
        pbar = comfy.utils.ProgressBar(len(imgs))
        for idx, img in enumerate(imgs):
            self.__last.append(b64_2_tensor(img))
            pbar.update_absolute(idx)
        return (self.__last, )
