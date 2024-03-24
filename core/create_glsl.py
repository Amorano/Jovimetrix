"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation - GLSL
"""

from enum import Enum

import torch
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import comfy_message, load_help, parse_reset, JOVImageMultiple, \
    WILDCARD, ROOT, JOV_GLSL
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_parameter, zip_longest_fill, \
    EnumConvertType
from Jovimetrix.sup.image import  cv2tensor_full, \
    pil2cv, tensor2pil, MIN_IMAGE_SIZE
from Jovimetrix.sup.shader import GLSL, CompileException

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX GLSL 🔺🟩🔵"
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

class GLSLNode(JOVImageMultiple):
    NAME = "GLSL (JOV) 🍩"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    HELP_URL = f"CREATE#-glsl"
    DESC = ""
    DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)

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
        batch = parse_parameter(Lexicon.BATCH, kw, (1, 30), clip_min=1)
        fragment = kw.get(Lexicon.FRAGMENT, [DEFAULT_FRAGMENT])
        param = kw.get(Lexicon.PARAM, [{}])
        wihi = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), clip_min=1)
        pA = parse_parameter(kw.get(Lexicon.PIXEL, None))
        hold = kw.get(Lexicon.WAIT, [False])
        reset = kw.get(Lexicon.RESET, [False])
        params = [tuple(x) for x in zip_longest_fill(batch, fragment, param, wihi, pA, hold, reset)]
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

class GLSLBaseNode(JOVImageMultiple):
    CATEGORY = JOV_CATEGORY
    HELP_URL = f"GLSL#-"
    # DESCRIPTION = load_help(NAME, CATEGORY, DESC, HELP_URL)
    FRAGMENT = ".glsl"

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__program = None
        self.__glsl = None

    def run(self, **kw) -> list[torch.Tensor]:
        pA = parse_parameter(kw.get(Lexicon.PIXEL_A, None))
        pB = parse_parameter(kw.get(Lexicon.PIXEL_B, None))
        wihi = parse_parameter(Lexicon.WH, kw, (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), clip_min=1)
        kw.pop(Lexicon.WH, None)
        frag = kw.pop("frag", [self.FRAGMENT])
        # clear any junk, since the rest are 'params'
        for x in ['param', 'iChannel0', 'iChannel1', 'iChannel2', 'iPosition',
                  'fragCoord', 'iResolution', 'iTime', 'iTimeDelta', 'iFrameRate',
                  'iFrame', 'fragColor', 'texture1', 'texture2', 'texture3']:
            kw.pop(x, None)

        images = []
        params = [tuple(x) for x in zip_longest_fill(pA, pB, wihi, frag)]
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, wihi, frag) in enumerate(params):
            param = {}
            for k, v in kw.items():
                v = v[idx]
                if type(v) == dict:
                    c = 0
                    val = []
                    while (p := v.get(str(c), None)) is not None:
                        val.append(float(p))
                        c += 1
                    v = tuple(val)
                param[k] = v

            width, height = wihi
            if pA is not None:
                pA = tensor2pil(pA)
                width, height = pA.size
            if self.__glsl is None or self.__program is None or self.__program != frag:
                if self.__glsl is not None:
                    self.__glsl = None
                self.__glsl = GLSL(frag, width, height, param)
                self.__program = frag
            else:
               self.__glsl.width = width
               self.__glsl.height = height

            image = self.__glsl.render(pA, param)
            image = pil2cv(image)
            images.append(cv2tensor_full(image))
            pbar.update_absolute(idx)
        return list(zip(*images))

class GLSLSelectRange(GLSLBaseNode):
    NAME = "SELECT RANGE GLSL (JOV)"
    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)
    FRAGMENT = str(JOV_GLSL / "clr" / "clr-flt-range.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.START: ("VEC3", {"default": (0., 0., 0.), "step": 0.01, "precision": 4,
                                     "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
            Lexicon.END: ("VEC3", {"default": (1., 1., 1.), "step": 0.01, "precision": 4,
                                    "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["start"] = kw.pop(Lexicon.START, [(0., 0., 0.)])
        kw["end"] = kw.pop(Lexicon.END, [(1., 1., 1.)])
        return super().run(**kw)

class GLSLColorGrayscale(GLSLBaseNode):
    NAME = "GRAYSCALE GLSL (JOV)"
    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)

    FRAGMENT = str(JOV_GLSL / "clr" / "clr-grayscale.glsl")
    DEFAULT = (0.299, 0.587, 0.114)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.RGB: ("VEC3", {"default": cls.DEFAULT, "step": 0.01, "precision": 4,
                                   "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        rgb = kw.pop(Lexicon.RGB, self.DEFAULT)
        kw["conversion"] = rgb
        return super().run(**kw)

class GLSLCreateNoise(GLSLBaseNode):
    NAME = "NOISE GLSL (JOV)"
    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)

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
        frags = []
        typ = kw.pop(Lexicon.TYPE, [EnumNoiseType.VALUE])
        for t in typ:
            frag = None
            match EnumNoiseType[t]:
                case EnumNoiseType.BROWNIAN:
                    frag = JOV_GLSL / "cre"/ "cre-nse-brownian.glsl"
                case EnumNoiseType.GRADIENT:
                    frag = JOV_GLSL / "cre"/ "cre-nse-gradient.glsl"
                case EnumNoiseType.MOSAIC:
                    frag = JOV_GLSL / "cre"/ "cre-nse-mosaic.glsl"
                case EnumNoiseType.PERLIN_2D:
                    frag = JOV_GLSL / "cre"/ "cre-nse-perlin_2D.glsl"
                case EnumNoiseType.SIMPLEX_2D:
                    frag = JOV_GLSL / "cre"/ "cre-nse-simplex_2D.glsl"
                case EnumNoiseType.VALUE:
                    frag = JOV_GLSL / "cre"/ "cre-nse-value.glsl"
            frags.append(frag)

        kw["frag"] = frags if len(frags) > 0 else [None]
        kw["seed"] = kw.pop(Lexicon.SEED, [0])
        return super().run(**kw)

class GLSLCreatePattern(GLSLBaseNode):
    NAME = "PATTERN GLSL (JOV)"
    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.TYPE: (EnumPatternType._member_names_, {"default": EnumPatternType.CHECKER.name}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["frag"] = None
        typ = kw.pop(Lexicon.TYPE, EnumPatternType.CHECKER)
        match EnumPatternType[typ]:
            case EnumPatternType.CHECKER:
                kw["frag"] = JOV_GLSL / "cre"/ "cre-pat-checker.glsl"

        return super().run(**kw)

class GLSLCreatePolygon(GLSLBaseNode):
    NAME = "POLYGON GLSL (JOV)"
    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)

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
        kw["sides"] = kw.pop(Lexicon.VALUE, 3)
        kw["radius"] = 1. / kw.pop(Lexicon.RADIUS, 1)
        return super().run(**kw)

class GLSLMap(GLSLBaseNode):
    NAME = "MAP GLSL (JOV)"
    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.TYPE: (EnumMappingType._member_names_, {"default": EnumMappingType.POLAR.name}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        frag = None
        typ = kw.pop(Lexicon.TYPE, EnumMappingType.POLAR)
        match EnumMappingType[typ]:
            case EnumMappingType.MERCATOR:
                frag = JOV_GLSL / "map"/ "map-mercator.glsl"
            case EnumMappingType.POLAR:
                frag = JOV_GLSL / "map"/ "map-polar.glsl"
            case EnumMappingType.RECT_EQUAL:
                frag = JOV_GLSL / "map"/ "map-rect_equal.glsl"

        kw["frag"] = str(frag) if frag is not None else frag
        kw["flip"] = kw.pop(Lexicon.FLIP, False)
        return super().run(**kw)

class GLSLTRSMirror(GLSLBaseNode):
    NAME = "MIRROR GLSL (JOV)"

    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)

    FRAGMENT = str(JOV_GLSL / "trs" / "trs-mirror.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "step": 0.01}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        center = parse_parameter(Lexicon.PIVOT, kw, (0.5, 0.5,), EnumConvertType.FLOAT,  0, 1)[0]
        kw["angle"] = -kw.pop(Lexicon.ANGLE, 0)
        kw["center"] = center
        return super().run(**kw)

class GLSLTRSRotate(GLSLBaseNode):
    NAME = "ROTATE GLSL (JOV)"

    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-rotate.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "step": 0.01}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        center = parse_parameter(Lexicon.PIVOT, kw, (0.5, 0.5,), EnumConvertType.FLOAT,  0, 1)[0]
        kw["angle"] = -kw.pop(Lexicon.ANGLE, 0)
        kw["center"] = center
        return super().run(**kw)

class GLSLUtilTiler(GLSLBaseNode):
    NAME = "TILER GLSL (JOV)"

    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-tiler.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.TILE: ("VEC2", {"default": (1., 1., ), "step": 0.1, "precision": 4,
                                     "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["uTile"] = parse_parameter(Lexicon.TILE, kw, (1., 1.,), EnumConvertType.FLOAT,
                                  1)
        kw.pop(Lexicon.TILE)
        return super().run(**kw)

class GLSLVFX(GLSLBaseNode):
    NAME = "VFX GLSL (JOV)"

    HELP_URL = f"GLSL#-"
    DESC = ""
    DESCRIPTION = load_help(NAME, JOV_CATEGORY, DESC, HELP_URL)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            "radius": ("FLOAT", {"default": 2., "min": 0.0001, "step": 0.01}),
            "strength": ("FLOAT", {"default": 1., "min": 0., "step": 0.01}),
            "center": ("VEC2", {"default": (0.5, 0.5, ), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.TYPE: (EnumVFXType._member_names_, {"default": EnumVFXType.BULGE.name})
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> list[torch.Tensor]:
        kw["frag"] = None
        typ = kw.pop(Lexicon.TYPE, EnumVFXType.BULGE)
        match EnumVFXType[typ]:
            case EnumVFXType.BULGE:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-bulge.glsl")

            case EnumVFXType.CHROMATIC:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-chromatic.glsl")

            case EnumVFXType.CROSS_HATCH:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-cross_hatch.glsl")

            case EnumVFXType.CRT:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-crt.glsl")

            case EnumVFXType.FILM_GRAIN:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-film-grain.glsl")

            case EnumVFXType.FROSTED:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-frosted.glsl")

            case EnumVFXType.PIXELATION:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-pixelation.glsl")

            case EnumVFXType.SEPIA:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-sepia.glsl")

            case EnumVFXType.VHS:
                kw["frag"] = str(JOV_GLSL / "vfx" / "vfx-vhs.glsl")

        return super().run(**kw)
