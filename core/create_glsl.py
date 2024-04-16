"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation - GLSL
"""

from enum import Enum

import torch
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOV_WEB_RES_ROOT, comfy_message, parse_reset, JOVBaseNode, \
    WILDCARD, ROOT, JOV_GLSL
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_list_value, zip_longest_fill, \
    EnumConvertType
from Jovimetrix.sup.image import  cv2tensor_full, \
    pil2cv, tensor2pil, MIN_IMAGE_SIZE
from Jovimetrix.sup.shader import GLSL, CompileException

# =============================================================================

JOV_CATEGORY = "GLSL"
JOV_CONFIG_GLSL = ROOT / 'glsl'
DEFAULT_FRAGMENT = """void main() {
    vec4 texColor = texture(iChannel0, fragCoord);
    vec4 color = vec4(fragCoord, abs(sin(iTime)), 1.0);
    fragColor = vec4((texColor.xyz + color.xyz) / 2.0, 1.0);
}"""

# =============================================================================

class EnumMappingType(Enum):
    MERCATOR = 10
    POLAR = 20
    RECT_EQUAL = 30

class EnumVFXType(Enum):
    BULGE = 10
    CHROMATIC = 20
    CROSS_STITCH = 30
    CROSS_HATCH = 40
    CRT = 50
    FILM_GRAIN = 60
    FROSTED = 70
    PIXELATION = 80
    SEPIA = 90
    VHS = 100

class EnumNoiseType(Enum):
    BROWNIAN = 0
    VALUE = 10
    GRADIENT = 20
    MOSAIC = 30
    PERLIN_2D = 40
    #PERLIN_2D_P = 42
    SIMPLEX_2D = 50
    #SIMPLEX_3D = 52

class EnumPatternType(Enum):
    CHECKER = 10

# =============================================================================

class GLSLNode(JOVBaseNode):
    NAME = "GLSL (JOV) 🍩"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.TIME: ("FLOAT", {"default": 0, "step": 0.0001, "min": 0, "precision": 6}),
            Lexicon.BATCH: ("VEC2", {"default": (1, 30), "step": 1, "label": ["COUNT", "FPS"]}),
            Lexicon.WAIT: ("BOOLEAN", {"default": False}),
            Lexicon.RESET: ("BOOLEAN", {"default": False}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1,}),
            Lexicon.FRAGMENT: ("STRING", {"multiline": True, "default": DEFAULT_FRAGMENT, "dynamicPrompts": False}),
            Lexicon.PARAM: ("STRING", {"default": {}})
        },
        "hidden": {
            "ident": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__glsl = None
        self.__fragment = ""
        self.__last_good = [torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=torch.uint8, device="cpu")]

    def run(self, ident, **kw) -> list[torch.Tensor]:
        batch = parse_list_value(kw.get(Lexicon.BATCH, None), EnumConvertType.VEC2INT, 1, (1, 30))
        fragment = parse_list_value(kw.get(Lexicon.FRAGMENT, None), EnumConvertType.STRING, DEFAULT_FRAGMENT)
        param = parse_list_value(kw.get(Lexicon.PARAM, None), EnumConvertType.DICT, {})
        wihi = parse_list_value(kw.get(Lexicon.WH, None), EnumConvertType.VEC2INT, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), MIN_IMAGE_SIZE)
        pA = parse_list_value(kw.get(Lexicon.PIXEL, None), EnumConvertType.IMAGE, None)
        hold = parse_list_value(kw.get(Lexicon.WAIT, None), EnumConvertType.BOOLEAN, False)
        reset = parse_list_value(kw.get(Lexicon.RESET, None), EnumConvertType.BOOLEAN, False)
        params = zip_longest_fill(batch, fragment, param, wihi, pA, hold, reset)
        images = []
        pbar = ProgressBar(len(params))
        for idx, (batch, fragment, param, wihi, pA, hold, reset) in enumerate(params):
            width, height = wihi
            batch_size, batch_fps = batch
            if self.__fragment != fragment or self.__glsl is None:
                try:
                    self.__glsl = GLSL(fragment, width, height, param)
                except CompileException as e:
                    comfy_message(ident, "jovi-glsl-error", {"id": ident, "e": str(e)})
                    logger.error(e)
                    return (self.__last_good, )
                self.__fragment = fragment

            if width != self.__glsl.width:
                self.__glsl.width = width
            if height != self.__glsl.height:
                self.__glsl.height = height
            pA = tensor2pil(pA) if pA is not None else None
            self.__glsl.hold = hold
            if parse_reset(ident) > 0 or reset:
                self.__glsl.reset()
                # comfy_message(ident, "jovi-glsl-time", {"id": ident, "t": 0})

            self.__glsl.fps = batch_fps
            for _ in range(batch_size):
                img = self.__glsl.render(pA, param)
                images.append(cv2tensor_full(pil2cv(img)))
            runtime = self.__glsl.runtime if not reset else 0
            comfy_message(ident, "jovi-glsl-time", {"id": ident, "t": runtime})

            self.__last_good = images
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class GLSLBaseNode(JOVBaseNode):
    NAME = ""
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    FRAGMENT = ".glsl"
    SORT = 100

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__program = None
        self.__glsl = None

    def run(self, **kw) -> list[torch.Tensor]:
        pA = parse_list_value(kw.get(Lexicon.PIXEL_A, None), EnumConvertType.IMAGE, None)
        kw.pop(Lexicon.PIXEL_A, None)
        pB = parse_list_value(kw.get(Lexicon.PIXEL_B, None), EnumConvertType.IMAGE, None)
        kw.pop(Lexicon.PIXEL_B, None)
        wihi = parse_list_value(kw.get(Lexicon.WH, None), EnumConvertType.VEC2INT, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), MIN_IMAGE_SIZE)
        kw.pop(Lexicon.WH, None)
        kw.pop(Lexicon.FRAGMENT, None)
        # clear any junk, since the rest are 'params'
        for x in ['param', 'iChannel0', 'iChannel1', 'iChannel2', 'iPosition',
                  'fragCoord', 'iResolution', 'iTime', 'iTimeDelta', 'iFrameRate',
                  'iFrame', 'fragColor', 'texture1', 'texture2', 'texture3']:
            kw.pop(x, None)

        images = []
        params = zip_longest_fill(pA, pB, wihi)
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, wihi) in enumerate(params):
            param = {k: v[idx] for k, v in kw.items()}
            width, height = wihi
            if pA is not None:
                pA = tensor2pil(pA)
                width, height = pA.size
            if self.__glsl is None or self.__program is None or self.__program != self.FRAGMENT:
                if self.__glsl is not None:
                    self.__glsl = None
                try:
                    self.__glsl = GLSL(self.FRAGMENT, width, height, param)
                    self.__program = self.FRAGMENT
                except CompileException as e:
                    logger.error(e)
                    logger.warning(param)
                    logger.warning(self.FRAGMENT)
            else:
               self.__glsl.width = width
               self.__glsl.height = height

            image = self.__glsl.render(pA, param)
            image = pil2cv(image)
            images.append(cv2tensor_full(image))
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

# =============================================================================

class GLSLSelectRange(GLSLBaseNode):
    NAME = "SELECT RANGE GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    FRAGMENT = str(JOV_GLSL / "clr" / "clr-flt-range.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.START: ("VEC3", {"default": (0., 0., 0.), "step": 0.01, "precision": 4,
                                     "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
            Lexicon.END: ("VEC3", {"default": (1., 1., 1.), "step": 0.01, "precision": 4,
                                    "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["start"] = parse_list_value(kw.get(Lexicon.START, None), EnumConvertType.VEC3, (0, 0, 0), 0, 1)
        kw.pop(Lexicon.START, None)
        kw["end"] = parse_list_value(kw.get(Lexicon.END, None), EnumConvertType.VEC3, (1, 1, 1), 0, 1)
        kw.pop(Lexicon.END, None)
        return super().run(**kw)

class GLSLColorGrayscale(GLSLBaseNode):
    NAME = "GRAYSCALE GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    FRAGMENT = str(JOV_GLSL / "clr" / "clr-grayscale.glsl")
    DEFAULT = (0.299, 0.587, 0.114)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.RGB: ("VEC3", {"default": cls.DEFAULT, "step": 0.01, "precision": 4,
                                   "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["conversion"] = parse_list_value(kw.get(Lexicon.RGB, None), EnumConvertType.VEC3, self.DEFAULT, 0, 1)
        kw.pop(Lexicon.RGB, None)
        return super().run(**kw)

class GLSLCreateNoise(GLSLBaseNode):
    NAME = "NOISE GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.TYPE: (EnumNoiseType._member_names_, {"default": EnumNoiseType.VALUE.name}),
            Lexicon.SEED: ("INT", {"default": 0, "step": 1}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1,
                                   "label": [Lexicon.W, Lexicon.H]})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw[Lexicon.FRAGMENT] = []
        typ = parse_list_value(kw.get(Lexicon.TYPE, None), EnumConvertType.STRING, EnumNoiseType.VALUE.name)
        kw.pop(Lexicon.TYPE, None)
        for t in typ:
            match EnumNoiseType[t]:
                case EnumNoiseType.BROWNIAN:
                    frag = JOV_GLSL / "cre"/ "cre-nse-brownian.glsl"
                case EnumNoiseType.GRADIENT:
                    frag = JOV_GLSL / "cre"/ "cre-nse-gradient.glsl"
                case EnumNoiseType.MOSAIC:
                    frag = JOV_GLSL / "cre"/ "cre-nse-mosaic.glsl"
                case EnumNoiseType.PERLIN_2D:
                    frag = JOV_GLSL / "cre"/ "cre-nse-perlin.glsl"
                case EnumNoiseType.SIMPLEX_2D:
                    frag = JOV_GLSL / "cre"/ "cre-nse-simplex.glsl"
                case EnumNoiseType.VALUE:
                    frag = JOV_GLSL / "cre"/ "cre-nse-value.glsl"
            kw[Lexicon.FRAGMENT].append(frag)
        kw["seed"] = parse_list_value(kw.get(Lexicon.SEED, None), EnumConvertType.INT, 0)
        kw.pop(Lexicon.SEED, None)
        return super().run(**kw)

class GLSLCreatePattern(GLSLBaseNode):
    NAME = "PATTERN GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.TYPE: (EnumPatternType._member_names_, {"default": EnumPatternType.CHECKER.name}),
            Lexicon.TILE: ("VEC2", {"default": (1, 1), "step": 0.02, "precision": 6, "min": 1, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["uTile"] = parse_list_value(kw.get(Lexicon.TILE, None), EnumConvertType.VEC2, (1, 1), 1)
        kw.pop(Lexicon.TILE)
        kw[Lexicon.FRAGMENT] = []
        typ = parse_list_value(kw.get(Lexicon.TYPE, None), EnumConvertType.STRING, EnumPatternType.CHECKER.name)
        kw.pop(Lexicon.TYPE, None)
        for t in typ:
            match EnumPatternType[t]:
                case EnumPatternType.CHECKER:
                    val = JOV_GLSL / "cre"/ "cre-pat-checker.glsl"
            kw[Lexicon.FRAGMENT].append(val)
        return super().run(**kw)

class GLSLCreatePolygon(GLSLBaseNode):
    NAME = "POLYGON GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    FRAGMENT = str(JOV_GLSL / "cre" / "cre-shp-polygon.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.VALUE: ("INT", {"default": 3, "step": 1, "min": 3}),
            Lexicon.RADIUS: ("FLOAT", {"default": 1, "min": 0.01, "max": 4, "step": 0.01}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["sides"] = parse_list_value(kw.get(Lexicon.VALUE, None), EnumConvertType.INT, 3, 3)
        kw.pop(Lexicon.VALUE, None)
        val = parse_list_value(kw.get(Lexicon.RADIUS, None), EnumConvertType.FLOAT, 1, 1)
        kw["radius"] = [1. / v for v in val]
        kw.pop(Lexicon.RADIUS, None)
        return super().run(**kw)

class GLSLMap(GLSLBaseNode):
    NAME = "MAP GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.TYPE: (EnumMappingType._member_names_, {"default": EnumMappingType.POLAR.name}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw[Lexicon.FRAGMENT] = []
        typ = parse_list_value(kw.get(Lexicon.TYPE, None), EnumConvertType.STRING, EnumMappingType.POLAR.name)
        kw.pop(Lexicon.TYPE, None)
        for t in typ:
            match EnumMappingType[t]:
                case EnumMappingType.MERCATOR:
                    f = JOV_GLSL / "map"/ "map-mercator.glsl"
                case EnumMappingType.POLAR:
                    f = JOV_GLSL / "map"/ "map-polar.glsl"
                case EnumMappingType.RECT_EQUAL:
                    f = JOV_GLSL / "map"/ "map-rect_equal.glsl"
            kw[Lexicon.FRAGMENT].append(f)
        kw["flip"] = parse_list_value(kw.get(Lexicon.FLIP, None), EnumConvertType.BOOLEAN, False)
        kw.pop(Lexicon.FLIP, None)
        return super().run(**kw)

class GLSLTRSMirror(GLSLBaseNode):
    NAME = "MIRROR GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-mirror.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "step": 0.01}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["center"] = parse_list_value(kw.get(Lexicon.PIVOT, None), EnumConvertType.VEC2, (0.5, 0.5), 0, 1)
        kw.pop(Lexicon.PIVOT, None)
        kw["uZoom"] = parse_list_value(kw.get(Lexicon.ANGLE, None), EnumConvertType.FLOAT, 0)
        kw["uZoom"] = [-a for a in kw["uZoom"]]
        kw.pop(Lexicon.ANGLE, None)

        return super().run(**kw)

class GLSLTRSRotate(GLSLBaseNode):
    NAME = "ROTATE GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-rotate.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "step": 0.01}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["center"] = parse_list_value(kw.get(Lexicon.PIVOT, None), EnumConvertType.VEC2, (0.5, 0.5), 0, 1)
        kw.pop(Lexicon.PIVOT, None)
        kw["angle"] = parse_list_value(kw.get(Lexicon.ANGLE, None), EnumConvertType.FLOAT, 0)
        kw["angle"] = [-a for a in kw["angle"]]
        kw.pop(Lexicon.ANGLE, None)
        return super().run(**kw)

class GLSLUtilTiler(GLSLBaseNode):
    NAME = "TILER GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-tiler.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.TILE: ("VEC2", {"default": (1., 1., ), "step": 0.1, "precision": 4,
                                     "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["uTile"] = parse_list_value(kw.get(Lexicon.TILE, None), EnumConvertType.VEC2, (1, 1), 1)
        kw.pop(Lexicon.TILE)
        return super().run(**kw)

class GLSLTRSKaleidoscope(GLSLBaseNode):
    NAME = "KALEIDOSCOPE GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-kaleidoscope.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.SEGMENT: ("FLOAT", {"default": 2.5, "step": 0.002, "min": 2,
                                        "precision": 6}),
            Lexicon.RADIUS: ("FLOAT", {"default": 1, "step": 0.002, "min": 0.00001,
                                       "max": 2.0, "precision": 6}),
            Lexicon.ZOOM: ("FLOAT", {"default": 1, "step": 0.002, "min": 0.00001,
                                      "max": 2.0, "precision": 6, "tooltip":"regression value"}),
            Lexicon.OFFSET: ("VEC2", {"default": (0.5, 0.5), "step": 0.005, "precision": 6,
                                      "min": 0, "max": 1,
                                      "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.ROTATE: ("FLOAT", {"default": 0, "step": 0.002, "precision": 6,
                                       "min": -1.7976931348623157e+308, "max": 1.7976931348623157e+308,
                                       "tooltip":"spin the input texture for each segment"}),
            Lexicon.SIZE: ("FLOAT", {"default": 0.5, "step": 0.002, "precision": 6,
                                     "min": 0.00001, "max": 10}),
            Lexicon.SKIP: ("FLOAT", {"default": 0, "step": 0.002, "precision": 6,
                                     "min": -0.5, "max": 0.5}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["segments"] = parse_list_value(kw.get(Lexicon.SEGMENT, None), EnumConvertType.FLOAT, 2, 2.5)
        kw.pop(Lexicon.SEGMENT)
        kw["radius"] = parse_list_value(kw.get(Lexicon.RADIUS, None), EnumConvertType.FLOAT, 1, 0.00001, 2)
        kw.pop(Lexicon.RADIUS)
        kw["regress"] = parse_list_value(kw.get(Lexicon.ZOOM, None), EnumConvertType.FLOAT, 1, 0, 2)
        kw.pop(Lexicon.ZOOM)
        kw["shift"] = parse_list_value(kw.get(Lexicon.OFFSET, None), EnumConvertType.VEC2, (0.5, 0.5), 0, 1)
        kw.pop(Lexicon.OFFSET)
        kw["spin"] = parse_list_value(kw.get(Lexicon.ROTATE, None), EnumConvertType.FLOAT, 0)
        kw.pop(Lexicon.ROTATE)
        kw["scale"] = parse_list_value(kw.get(Lexicon.SIZE, None), EnumConvertType.FLOAT, 0.5, 0.00001, 10)
        kw.pop(Lexicon.SIZE)
        kw["skip"] = parse_list_value(kw.get(Lexicon.SKIP, None), EnumConvertType.FLOAT, 0, 0.5, -0.5)
        kw.pop(Lexicon.SKIP)
        return super().run(**kw)

class GLSLVFX(GLSLBaseNode):
    NAME = "VFX GLSL (JOV)"
    NAME_URL = NAME.split(" (JOV)")[0].replace(" ", "%20")
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"{JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            "radius": ("FLOAT", {"default": 2., "min": 0.0001, "step": 0.01}),
            "strength": ("FLOAT", {"default": 1., "min": 0., "step": 0.01}),
            "center": ("VEC2", {"default": (0.5, 0.5, ), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.TYPE: (EnumVFXType._member_names_, {"default": EnumVFXType.BULGE.name})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw[Lexicon.FRAGMENT] = []
        typ = parse_list_value(kw.get(Lexicon.TYPE, None), EnumConvertType.STRING, EnumVFXType.BULGE.name)
        for t in typ:
            match EnumVFXType[t]:
                case EnumVFXType.BULGE:
                    val = str(JOV_GLSL / "vfx" / "vfx-bulge.glsl")

                case EnumVFXType.CHROMATIC:
                    val = str(JOV_GLSL / "vfx" / "vfx-chromatic.glsl")

                case EnumVFXType.CROSS_HATCH:
                    val = str(JOV_GLSL / "vfx" / "vfx-cross_hatch.glsl")

                case EnumVFXType.CRT:
                    val = str(JOV_GLSL / "vfx" / "vfx-crt.glsl")

                case EnumVFXType.FILM_GRAIN:
                    val = str(JOV_GLSL / "vfx" / "vfx-film-grain.glsl")

                case EnumVFXType.FROSTED:
                    val = str(JOV_GLSL / "vfx" / "vfx-frosted.glsl")

                case EnumVFXType.PIXELATION:
                    val = str(JOV_GLSL / "vfx" / "vfx-pixelation.glsl")

                case EnumVFXType.SEPIA:
                    val = str(JOV_GLSL / "vfx" / "vfx-sepia.glsl")

                case EnumVFXType.VHS:
                    val = str(JOV_GLSL / "vfx" / "vfx-vhs.glsl")
            kw[Lexicon.FRAGMENT].append(val)
        return super().run(**kw)
