"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from enum import Enum

import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager

from Jovimetrix import pil2tensor, pil2cv, cv2pil, cv2tensor, cv2mask, \
    deep_merge_dict, parse_tuple, parse_number, b64_2_image, b64_2_tensor, \
    EnumTupleType, JOVBaseNode, JOVImageBaseNode, Logger, Lexicon, \
    IT_PIXELS, IT_RGBA, IT_WH, IT_SCALE, IT_ROT, IT_INVERT, \
    IT_TIME, IT_WHMODE, IT_REQUIRED, MIN_IMAGE_SIZE

from Jovimetrix.sup.comp import geo_scalefit, shape_ellipse, channel_solid, \
    shape_polygon, shape_quad, light_invert, image_load_from_url, \
    EnumImageType, EnumInterpolation, EnumScaleMode, \
    IT_SAMPLE

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
        match shape:
            case EnumShapes.SQUARE:
                img = shape_quad(width, height, sizeX, sizeX, fill=color, back=bgcolor)

            case EnumShapes.ELLIPSE:
                img = shape_ellipse(width, height, sizeX, sizeY, fill=color, back=bgcolor)

            case EnumShapes.RECTANGLE:
                img = shape_quad(width, height, sizeX, sizeY, fill=color, back=bgcolor)

            case EnumShapes.POLYGON:
                img = shape_polygon(width, height, sizeX, sides, fill=color, back=bgcolor)

            case EnumShapes.CIRCLE:
                img = shape_ellipse(width, height, sizeX, sizeX, fill=color, back=bgcolor)

        img = img.rotate(-angle)
        if i != 0:
            img = pil2cv(img)
            img = light_invert(img, i)
            img = cv2pil(img)

        return (pil2tensor(img), pil2tensor(img.convert("L")), )

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
    # OUTPUT_IS_LIST = (False, False, )
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS) #, IT_TIME, d, IT_WH)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        Logger.debug(self, kw)
        pixels = kw.get(Lexicon.PIXEL, [None])
        # load_base64
        #wh = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE))
        #image = Image.new(mode="RGB", size=wh[0])
        #return (pil2tensor(image), pil2mask(image))
        return (pixels, pixels, )

class LoadImageNode(JOVImageBaseNode):
    NAME = "IMAGE FROM URL (JOV) 📥"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d =  {"optional": {
            Lexicon.URL: ("STRING", {"default": "https://images.squarespace-cdn.com/content/v1/600743194a20ea052ee04fbb/9e868816-2c5b-4ba2-b157-9a0911dfc4c2/SNAILS_square3.jpg"}),
        }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WHMODE, IT_SAMPLE)

    def __init__(self) -> None:
        self.__url = None
        self.__image = None

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        url = kw[Lexicon.URL]
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        if self.__url != url:
            self.__url = url
            self.__image = image_load_from_url(url)

        image = self.__image
        if image is None:
            image = channel_solid(width, height, 0, chan=EnumImageType.RGB)
        else:
            mode = kw.get(Lexicon.MODE, EnumScaleMode.NONE)
            mode = EnumScaleMode[mode]
            sample = kw.get(Lexicon.SAMPLE, EnumInterpolation.LANCZOS4)
            sample = EnumInterpolation[sample]
            image = geo_scalefit(image, width, height, mode, sample)
        return (cv2tensor(image), cv2mask(image),)
