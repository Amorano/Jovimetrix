"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from enum import Enum

import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager
from loguru import logger

import comfy
from server import PromptServer

from Jovimetrix import ComfyAPIMessage, JOVBaseNode, JOVImageBaseNode, \
    ROOT, IT_PIXELS, IT_RGBA, IT_WH, IT_SCALE, IT_ROT, IT_INVERT, \
    IT_REQUIRED, MIN_IMAGE_SIZE, TimedOutException

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import deep_merge_dict, parse_tuple, parse_number, \
    EnumTupleType

from Jovimetrix.sup.image import channel_add, pil2tensor, pil2cv, \
    cv2tensor, cv2mask, IT_WHMODE, tensor2pil

from Jovimetrix.sup.comp import shape_ellipse, shape_polygon, shape_quad, \
    light_invert, EnumScaleMode

from Jovimetrix.sup.shader import GLSL, CompileException

JOV_CONFIG_GLSL = ROOT / 'glsl'

FONT_MANAGER = matplotlib.font_manager.FontManager()
FONTS = {font.name: font.fname for font in FONT_MANAGER.ttflist}
FONT_NAMES = sorted(FONTS.keys())

DEFAULT_FRAGMENT = """void main() {
    vec4 texColor = texture(iChannel0, iUV);
    vec4 color = vec4(iUV, abs(sin(iTime)), 1.0);
    fragColor = vec4((texColor.xyz + color.xyz) / 2.0, 1.0);
}"""
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
    WIDTH = 512
    HEIGHT = 512

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.TIME: ("FLOAT", {"default": 0, "step": 0.0001, "min": 0, "precision": 6}),
                Lexicon.FPS: ("INT", {"default": 0, "step": 1, "min": 0, "max": 1000}),
                Lexicon.BATCH: ("INT", {"default": 1, "step": 1, "min": 1, "max": 36000}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
                Lexicon.RESET: ("BOOLEAN", {"default": False}),
                Lexicon.WH: ("VEC2", {"default": (cls.WIDTH, cls.HEIGHT,), "step": 1, "min": 1}),
                Lexicon.FRAGMENT: ("STRING", {"multiline": True, "default": DEFAULT_FRAGMENT, "dynamicPrompts": False}),
                Lexicon.PARAM: ("STRING", {"default": ""})
            },
            "hidden": {
                "id": "UNIQUE_ID"
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__glsl = None
        self.__fragment = ""
        self.__last_good = [torch.zeros((self.WIDTH, self.HEIGHT, 3), dtype=torch.uint8)]

    def run(self, id, **kw) -> list[torch.Tensor]:
        batch = kw.get(Lexicon.BATCH, 1)
        fragment = kw.get(Lexicon.FRAGMENT, DEFAULT_FRAGMENT)
        width, height = parse_tuple(Lexicon.WH, kw, default=(self.WIDTH, self.HEIGHT,), clip_min=1)[0]
        if self.__fragment != fragment or self.__glsl is None:
            try:
                self.__glsl = GLSL(fragment, width, height)
            except CompileException as e:
                PromptServer.instance.send_sync("jovi-glsl-error", {"id": id, "e": str(e)})
                logger.error(e)
                return (self.__last_good, )
            self.__fragment = fragment

        if width != self.__glsl.width:
            self.__glsl.width = width

        if height != self.__glsl.height:
            self.__glsl.height = height

        frames = []
        if (texture1 := kw.get(Lexicon.PIXEL, None)) is not None:
            texture1 = tensor2pil(texture1)

        if (texture2 := kw.get(Lexicon.PIXEL, None)) is not None:
            texture2 = tensor2pil(texture2)

        self.__glsl.hold = kw.get(Lexicon.WAIT, False)

        reset = kw.get(Lexicon.RESET, False)
        # clear the queue of msgs...
        # better resets? check if reset message
        try:
            data = ComfyAPIMessage.poll(id, timeout=0)
            # logger.debug(data)
            if (cmd := data.get('cmd', None)) is not None:
                if cmd == 'reset':
                    reset = True
        except TimedOutException as e:
            pass
        except Exception as e:
            logger.error(str(e))

        if reset:
            self.__glsl.reset()
            # PromptServer.instance.send_sync("jovi-glsl-time", {"id": id, "t": 0})

        self.__glsl.fps = kw.get(Lexicon.FPS, 0)

        pbar = comfy.utils.ProgressBar(batch)
        for idx in range(batch):
            img = self.__glsl.render(texture1, texture2)
            frames.append(pil2tensor(img))
            pbar.update_absolute(idx)

        runtime = self.__glsl.runtime if not reset else 0
        PromptServer.instance.send_sync("jovi-glsl-time", {"id": id, "t": runtime})

        self.__last_good = frames
        return (self.__last_good, )
