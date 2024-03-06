"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation - GLSL
"""

from enum import Enum

import torch
from loguru import logger

import comfy
from server import PromptServer

from Jovimetrix import JOV_HELP_URL, WILDCARD, ComfyAPIMessage, \
    JOVBaseNode, JOVImageMultiple, TimedOutException, \
    ROOT, MIN_IMAGE_SIZE, JOV_GLSL

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_tuple, zip_longest_fill, EnumTupleType
from Jovimetrix.sup.image import batch_extract, cv2tensor_full, pil2cv, pil2tensor, tensor2pil
from Jovimetrix.sup.shader import GLSL, CompileException

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
JOV_CONFIG_GLSL = ROOT / 'glsl'
DEFAULT_FRAGMENT = """void main() {
    vec4 texColor = texture(iChannel0, fragCoord);
    vec4 color = vec4(fragCoord, abs(sin(iTime)), 1.0);
    fragColor = vec4((texColor.xyz + color.xyz) / 2.0, 1.0);
}"""

# =============================================================================

class GLSLNode(JOVImageMultiple):
    NAME = "GLSL (JOV) 🍩"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = ""
    WIDTH = 512
    HEIGHT = 512

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
            Lexicon.WH: ("VEC2", {"default": (cls.WIDTH, cls.HEIGHT,), "step": 1,}),
            Lexicon.FRAGMENT: ("STRING", {"multiline": True, "default": DEFAULT_FRAGMENT, "dynamicPrompts": False}),
            Lexicon.PARAM: ("STRING", {"default": ""})
        },
        "hidden": {
            "id": "UNIQUE_ID"
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__glsl = None
        self.__fragment = ""
        self.__last_good = [torch.zeros((self.WIDTH, self.HEIGHT, 3), dtype=torch.uint8, device="cpu")]

    def run(self, id, **kw) -> list[torch.Tensor]:
        batch = parse_tuple(Lexicon.BATCH, kw, default=(1, 30), clip_min=1)
        fragment = kw.get(Lexicon.FRAGMENT, [DEFAULT_FRAGMENT])
        param = kw.get(Lexicon.PARAM, [{}])
        wihi = parse_tuple(Lexicon.WH, kw, default=(self.WIDTH, self.HEIGHT,), clip_min=1)
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
        hold = kw[Lexicon.WAIT]
        reset = kw[Lexicon.RESET]
        params = [tuple(x) for x in zip_longest_fill(batch, fragment, param, wihi, pA, hold, reset)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (batch, fragment, param, wihi, pA, hold, reset) in enumerate(params):
            width, height = wihi
            batch_size, batch_fps = batch
            if self.__fragment != fragment or self.__glsl is None:
                try:
                    self.__glsl = GLSL(fragment, width, height, param)
                except CompileException as e:
                    PromptServer.instance.send_sync("jovi-glsl-error", {"id": id, "e": str(e)})
                    logger.error(e)
                    return (self.__last_good, )
                self.__fragment = fragment

            if width != self.__glsl.width:
                self.__glsl.width = width
            if height != self.__glsl.height:
                self.__glsl.height = height
            pA = tensor2pil(pA) if pA is not None else None
            self.__glsl.hold = hold
            try:
                data = ComfyAPIMessage.poll(id, timeout=0)
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

            self.__glsl.fps = batch_fps
            for _ in range(batch_size):
                img = self.__glsl.render(pA, param)
                images.append(cv2tensor_full(pil2cv(img)))
            runtime = self.__glsl.runtime if not reset else 0
            PromptServer.instance.send_sync("jovi-glsl-time", {"id": id, "t": runtime})

            self.__last_good = images
            pbar.update_absolute(idx)
        return list(zip(*images))

class GLSLBaseNode(JOVBaseNode):
    CATEGORY = "JOVIMETRIX GLSL"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    FRAGMENT = ".glsl"

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__program = None
        self.__glsl = None

    def run(self, **kw) -> list[torch.Tensor]:
        width, height = MIN_IMAGE_SIZE, MIN_IMAGE_SIZE
        pA = kw.get(Lexicon.PIXEL, None)
        pA = None if pA is None else batch_extract(pA)[0]
        if pA is not None:
            pA = tensor2pil(pA)
            width, height = pA.size

        wihi = parse_tuple(Lexicon.WH, kw, default=(width, height,), clip_min=1)[0]
        width, height = wihi
        kw.pop(Lexicon.WH, None)
        seed = kw.pop(Lexicon.SEED, None)

        uv_tile = None
        if (uv_tile := kw.pop(Lexicon.TILE, None)) is not None:
            uv_tile = parse_tuple([uv_tile], typ=EnumTupleType.FLOAT, default=(1., 1.,), clip_min=0.01)[0]

        if (pB := kw.pop(Lexicon.PIXEL_B, None)) is not None:
            # texture2 = tensor2pil(texture2)
            pB = None if pB is None else batch_extract(pB)[0]

        #if (texture3 := kw.pop(Lexicon.MASK, None)) is not None:
        #    texture3 = tensor2pil(texture3)

        frag = kw.pop("frag", self.FRAGMENT)
        for x in ['param', 'iChannel0', 'iChannel1', 'iChannel2', 'iPosition', 'fragCoord', 'iResolution', 'iTime', 'iTimeDelta', 'iFrameRate', 'iFrame', 'fragColor', 'texture1', 'texture2', 'texture3']:
            kw.pop(x, None)

        param = {}
        for k, v in kw.items():
            if type(v) == dict:
                v = parse_tuple(k, kw, EnumTupleType.FLOAT)[0]
            param[k] = v

        if uv_tile is not None:
            param['uv_tile'] = uv_tile
        if seed is not None:
            param['seed'] = seed

        if self.__glsl is None or self.__program is None or self.__program != frag:
            self.__program = frag
            if self.__glsl is not None:
                del self.__glsl

            self.__glsl = None
            try:
                self.__glsl = GLSL(self.__program, width=width, height=height, param=param)
            except Exception as e:
                logger.error(str(e))
                logger.error(self.__program)
                ret = [torch.zeros((height, width, 3), dtype=torch.uint8, device="cpu")]
                return (ret, )

        self.__glsl.width = width
        self.__glsl.height = height
        img = self.__glsl.render(pA, param)
        return (pil2tensor(img), )

class GLSLSelectRange(GLSLBaseNode):
    NAME = "SELECT RANGE GLSL (JOV)"
    FRAGMENT = str(JOV_GLSL / "clr" / "clr-flt-range.glsl")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.START: ("VEC3", {"default": (0., 0., 0.), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
            Lexicon.END: ("VEC3", {"default": (1., 1., 1.), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    def run(self, **kw) -> list[torch.Tensor]:
        kw["start"] = kw.pop(Lexicon.START, (0., 0., 0.))
        kw["end"] = kw.pop(Lexicon.END, (1., 1., 1.))
        return super().run(**kw)

class GLSLColorGrayscale(GLSLBaseNode):
    NAME = "GRAYSCALE GLSL (JOV)"
    FRAGMENT = str(JOV_GLSL / "clr" / "clr-grayscale.glsl")
    DEFAULT = (0.299, 0.587, 0.114)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.RGB: ("VEC3", {"default": cls.DEFAULT, "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    def run(self, **kw) -> list[torch.Tensor]:
        rgb = kw.pop(Lexicon.RGB, self.DEFAULT)
        kw["conversion"] = rgb
        return super().run(**kw)

class EnumNoiseType(Enum):
    BROWNIAN = 0
    VALUE = 10
    GRADIENT = 20
    MOSAIC = 30
    PERLIN_2D = 40
    PERLIN_2D_P = 42
    SIMPLEX_2D = 50
    SIMPLEX_3D = 52

class GLSLCreateNoise(GLSLBaseNode):
    NAME = "NOISE GLSL (JOV)"
    CATEGORY = "JOVIMETRIX GLSL"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.TYPE: (EnumNoiseType._member_names_, {"default": EnumNoiseType.VALUE.name}),
            Lexicon.SEED: ("INT", {"default": 0, "step": 1}),
            Lexicon.TILE: ("VEC2", {"default": (1., 1.,), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.WH: ("VEC2", {"default": (512, 512,), "step": 1}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    def run(self, **kw) -> list[torch.Tensor]:
        frag = None
        typ = kw.pop(Lexicon.TYPE, EnumNoiseType.VALUE)
        match EnumNoiseType[typ]:
            case EnumNoiseType.BROWNIAN:
                frag = JOV_GLSL / "cre"/ "cre-nse-brownian.glsl"
            case EnumNoiseType.GRADIENT:
                frag = JOV_GLSL / "cre"/ "cre-nse-gradient.glsl"
            case EnumNoiseType.MOSAIC:
                frag = JOV_GLSL / "cre"/ "cre-nse-mosaic.glsl"
            case EnumNoiseType.PERLIN_2D:
                frag = JOV_GLSL / "cre"/ "cre-nse-perlin_2D.glsl"
            case EnumNoiseType.PERLIN_2D_P:
                frag = JOV_GLSL / "cre"/ "cre-nse-perlin_2D-periodic.glsl"
            case EnumNoiseType.SIMPLEX_2D:
                frag = JOV_GLSL / "cre"/ "cre-nse-simplex_2D.glsl"
            case EnumNoiseType.SIMPLEX_3D:
                frag = JOV_GLSL / "cre"/ "cre-nse-simplex_3D.glsl"
            case EnumNoiseType.VALUE:
                frag = JOV_GLSL / "cre"/ "cre-nse-value.glsl"

        kw["frag"] = str(frag) if frag is not None else None
        return super().run(**kw)

class EnumPatternType(Enum):
    CHECKER = 10

class GLSLCreatePattern(GLSLBaseNode):
    NAME = "PATTERN GLSL (JOV)"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.TYPE: (EnumPatternType._member_names_, {"default": EnumPatternType.CHECKER.name}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    def run(self, **kw) -> list[torch.Tensor]:
        kw["frag"] = None
        typ = kw.pop(Lexicon.TYPE, EnumPatternType.CHECKER)
        match EnumPatternType[typ]:
            case EnumPatternType.CHECKER:
                kw["frag"] = JOV_GLSL / "cre"/ "cre-pat-checker.glsl"

        return super().run(**kw)

class GLSLCreatePolygon(GLSLBaseNode):
    NAME = "POLYGON GLSL (JOV)"
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
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    def run(self, **kw) -> list[torch.Tensor]:
        kw["sides"] = kw.pop(Lexicon.VALUE, 3)
        kw["radius"] = 1. / kw.pop(Lexicon.RADIUS, 1)
        return super().run(**kw)

class EnumMappingType(Enum):
    MERCATOR = 10
    POLAR = 20
    RECT_EQUAL = 30

class GLSLMap(GLSLBaseNode):
    NAME = "MAP GLSL (JOV)"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.TYPE: (EnumMappingType._member_names_, {"default": EnumMappingType.POLAR.name}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

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
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    def run(self, **kw) -> list[torch.Tensor]:
        center = parse_tuple(Lexicon.PIVOT, kw, typ=EnumTupleType.FLOAT, default=(0.5, 0.5,), clip_min=0, clip_max=1)[0]
        kw["angle"] = -kw.pop(Lexicon.ANGLE, 0)
        kw["center"] = center
        return super().run(**kw)

class GLSLTRSRotate(GLSLBaseNode):
    NAME = "ROTATE GLSL (JOV)"
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
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    def run(self, **kw) -> list[torch.Tensor]:
        center = parse_tuple(Lexicon.PIVOT, kw, typ=EnumTupleType.FLOAT, default=(0.5, 0.5,), clip_min=0, clip_max=1)[0]
        kw["angle"] = -kw.pop(Lexicon.ANGLE, 0)
        kw["center"] = center
        return super().run(**kw)

class GLSLUtilTiler(GLSLBaseNode):
    NAME = "TILER GLSL (JOV)"
    FRAGMENT = JOV_GLSL / "utl" / "utl-tiler.glsl"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            "uTime": ("FLOAT", {"default": 0, "step": 0.01}),
            "uTile": ("VEC2", {"default": (1., 1., ), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

    def run(self, **kw) -> list[torch.Tensor]:
        uTime = kw.pop("uTime", 0.)
        uTile = parse_tuple("uTile", kw, typ=EnumTupleType.FLOAT, default=(1., 1.,), clip_min=1)[0]
        kw["uTime"] = uTime
        kw["uTile"] = uTile
        return super().run(**kw)

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

class GLSLVFX(GLSLBaseNode):
    NAME = "VFX GLSL (JOV)"

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
        return Lexicon._parse(d, JOV_HELP_URL + "/CREATE#-glsl")

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
