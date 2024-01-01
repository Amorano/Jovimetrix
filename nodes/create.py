"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from enum import Enum

import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager

from Jovimetrix import pil2tensor, pil2cv, cv2pil, cv2tensor, cv2mask, \
    deep_merge_dict, parse_tuple, parse_number,\
    EnumTupleType, JOVImageBaseNode, Logger, Lexicon, \
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
        text = kw[Lexicon.STRING]
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

"""
class PixelShaderNode(JOVImageInOutBaseNode):
    NAME = "PIXEL SHADER (JOV) 🔆"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.R: ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
                Lexicon.G: ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
                Lexicon.B: ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_WHMODE, IT_SAMPLE)

    @staticmethod
    def shader(image: TYPE_PIXEL, R: str, G: str, B: str, **kw) -> np.ndarray:

        from ast import literal_eval
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        out = np.zeros((height, width, 3), dtype=np.float32)
        R = R.strip()
        G = G.strip()
        B = B.strip()
        err = False

        for y in range(height):
            for x in range(width):
                variables = {
                    "$x": x,
                    "$y": y,
                    "$u": x / width if width > 0 else 0,
                    "$v": y / height if height > 0 else 0,
                    "$w": width,
                    "$h": height,
                    "$r": image[y, x, 2] / 255.,
                    "$g": image[y, x, 1] / 255.,
                    "$b": image[y, x, 0] / 255.,
                }

                parseR = re.sub(r'\$(\w+)', lambda match: str(variables.get(match.group(0), match.group(0))), R)
                parseG = re.sub(r'\$(\w+)', lambda match: str(variables.get(match.group(0), match.group(0))), G)
                parseB = re.sub(r'\$(\w+)', lambda match: str(variables.get(match.group(0), match.group(0))), B)

                for i, rgb in enumerate([parseB, parseG, parseR]):
                    if rgb == "":
                        out[y, x, i] = image[y, x, i]
                        continue

                    try:
                        out[y, x, i]  = literal_eval(rgb) * 255
                    except:
                        try:
                            out[y, x, i] = eval(rgb.replace("^", "**")) * 255
                        except Exception as e:
                            if not err:
                                err = True
                                Logger.err(f'eval failed {str(e)}\n{parseR}\n{parseG}\n{parseB}')

        return np.clip(out, 0, 255).astype(np.uint8)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        t = time.perf_counter()
        pixels = kw.get(Lexicon.PIXEL, [None])
        R = kw.get(Lexicon.R, [""])
        G = kw.get(Lexicon.G, [""])
        B = kw.get(Lexicon.B, [""])
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.FIT])
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, R, G, B, mode, sample):
            img, r, g, b, m, rs = data

            m = EnumScaleMode.FIT if m == EnumScaleMode.NONE else m
            # fix the image first -- must at least match for px, py indexes
            img = geo_scalefit(img, width, height, m, rs)
            if img is None:
                img = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                img = tensor2cv(img)
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height), interpolation=EnumInterpolation[rs])

            img = PixelShaderNode.shader(img, r, g, b)
            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        Logger.info(self.NAME, {time.perf_counter() - t:.5})
        return (
            torch.stack(images),
            torch.stack(masks)
        )
"""

class GLSLNode(JOVImageBaseNode):
    NAME = "GLSL (JOV) 🍩"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d =  {"optional": {
            Lexicon.RESET: ("BOOLEAN", {"default": True}),
            Lexicon.FRAGMENT: ("STRING", {"default":
"""void main() {
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    color.r += sin(iTime);
    FragColor = color;
}""",
                "multiline": True})
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_TIME, d, IT_WH)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        Logger.debug(self, kw)
        pixels = kw.get(Lexicon.PIXEL, [None])
        #wh = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE))
        #image = Image.new(mode="RGB", size=wh[0])
        #return (pil2tensor(image), pil2mask(image))
        return (pixels, pixels, )

class ImageFromURLNode(JOVImageBaseNode):
    NAME = "IMAGE FROM URL (JOV) 📥"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d =  {"optional": {
            Lexicon.URL: ("STRING", {"default": "https://upload.wikimedia.org/wikipedia/en/c/c0/Les_Horribles_Cernettes_in_1992.jpg"}),
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
            mode = EnumScaleMode[kw[Lexicon.MODE]]
            sample = EnumInterpolation[kw[Lexicon.SAMPLE]]
            image = geo_scalefit(image, width, height, mode, sample)
        return (cv2tensor(image), cv2mask(image),)
