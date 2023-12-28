"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from enum import Enum

import torch
from PIL import Image

from Jovimetrix import pil2tensor, pil2cv, cv2pil, cv2tensor, cv2mask, \
    deep_merge_dict, parse_tuple, parse_number,\
    EnumTupleType, JOVImageBaseNode, Logger, Lexicon, \
    IT_PIXELS, IT_RGBA, IT_WH, IT_SCALE, IT_ROT, IT_INVERT, \
    IT_TIME, IT_WHMODE, IT_REQUIRED, MIN_IMAGE_SIZE

from Jovimetrix.sup.comp import geo_scalefit, shape_ellipse, channel_solid, \
    shape_polygon, shape_quad, light_invert, image_load_from_url, \
    EnumInterpolation, EnumScaleMode, IT_SAMPLE

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
            }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WH, IT_RGBA, IT_ROT, IT_SCALE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        shape = kw.get(Lexicon.SHAPE, EnumShapes.CIRCLE)
        sides = kw.get(Lexicon.SIDES, 3)
        angle = kw.get(Lexicon.ANGLE, 0)
        sizeX, sizeY = parse_tuple(Lexicon.SIZE, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        color = parse_tuple(Lexicon.RGBA, kw, default=(255, 255, 255, 255), clip_min=1)[0]
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)[0]
        img = None
        match shape:
            case EnumShapes.SQUARE:
                img = shape_quad(width, height, sizeX, sizeX, fill=color)

            case EnumShapes.ELLIPSE:
                img = shape_ellipse(width, height, sizeX, sizeY, fill=color)

            case EnumShapes.RECTANGLE:
                img = shape_quad(width, height, sizeX, sizeY, fill=color)

            case EnumShapes.POLYGON:
                img = shape_polygon(width, height, sizeX, sides, fill=color)

            case EnumShapes.CIRCLE:
                img = shape_ellipse(width, height, sizeX, sizeX, fill=color)

        img = img.rotate(-angle)
        if (i or 0) > 0.:
            img = pil2cv(img)
            img = light_invert(img, i)
            img = cv2pil(img)

        return (pil2tensor(img), pil2tensor(img.convert("L")), )

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
        R = kw.get(Lexicon.R, [None])
        G = kw.get(Lexicon.G, [None])
        B = kw.get(Lexicon.B, [None])
        width, height = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, R, G, B, mode, sample):
            img, r, g, b, m, rs = data

            r = r if r is not None else ""
            g = g if g is not None else ""
            b = b if b is not None else ""
            m = m if m is not None else EnumScaleMode.FIT
            m = EnumScaleMode.FIT if m == EnumScaleMode.NONE else m

            # fix the image first -- must at least match for px, py indexes
            img = geo_scalefit(img, width, height, m, rs)

            if img is None:
                img = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                img = tensor2cv(img)
                if img.shape[0] != height or img.shape[1] != width:
                    s = EnumInterpolation.LANCZOS4
                    s = EnumInterpolation[rs] if rs is not None else s
                    img = cv2.resize(img, (width, height), interpolation=s)

            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
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
        pixels = kw.get(Lexicon.RESET, [None])
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
        default = (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,)
        width, height = parse_tuple(Lexicon.WH, kw, default=default, clip_min=1)[0]
        if self.__url != url:
            self.__url = url
            self.__image = image_load_from_url(url)

        if self.__image is None:
            self.__image = channel_solid(width, height, 0, chan=3)
        else:
            mode = EnumScaleMode[kw[Lexicon.MODE]]
            sample = EnumInterpolation[kw[Lexicon.SAMPLE]]
            self.__image = geo_scalefit(self.__image, width, height, mode, sample)
            print(width, height, self.__image.shape[:2], mode, sample)
        return (cv2tensor(self.__image), cv2mask(self.__image),)
