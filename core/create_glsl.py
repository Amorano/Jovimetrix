"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation - GLSL
"""

import sys
from enum import Enum
from typing import List

import torch
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import comfy_message, parse_reset, \
    JOVBaseNode, WILDCARD, ROOT, JOV_GLSL
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_param, zip_longest_fill, \
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
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    SORT = 1
    DESCRIPTION = """
The GLSL Node executes custom GLSL (OpenGL Shading Language) fragment shaders to generate images or apply effects. GLSL is a high-level shading language used for graphics programming, particularly in the context of rendering images or animations. This node allows for real-time rendering of shader effects, providing flexibility and creative control over image processing pipelines. It takes advantage of GPU acceleration for efficient computation, enabling the rapid generation of complex visual effects.
"""

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
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__glsl = None
        self.__fragment = ""
        self.__last_good = [torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=torch.uint8, device="cpu")]

    def run(self, ident, **kw) -> List[torch.Tensor]:
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.VEC2INT, [(1, 30)], 1)
        fragment = parse_param(kw, Lexicon.FRAGMENT, EnumConvertType.STRING, DEFAULT_FRAGMENT)
        param = parse_param(kw, Lexicon.PARAM, EnumConvertType.DICT, {})
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)], MIN_IMAGE_SIZE)
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        hold = parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False)
        reset = parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(batch, fragment, param, wihi, pA, hold, reset))
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
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    FRAGMENT = ".glsl"
    SORT = 100

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__program = None
        self.__glsl = None

    def run(self, **kw) -> List[torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        kw.pop(Lexicon.PIXEL_A, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        kw.pop(Lexicon.PIXEL_B, None)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)], MIN_IMAGE_SIZE)
        kw.pop(Lexicon.WH, None)
        kw.pop(Lexicon.FRAGMENT, None)
        # clear any junk, since the rest are 'params'
        for x in ['param', 'iChannel0', 'iChannel1', 'iChannel2', 'iPosition',
                  'fragCoord', 'iResolution', 'iTime', 'iTimeDelta', 'iFrameRate',
                  'iFrame', 'fragColor', 'texture1', 'texture2', 'texture3']:
            kw.pop(x, None)

        images = []
        params = list(zip_longest_fill(pA, pB, wihi))
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
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    FRAGMENT = str(JOV_GLSL / "clr" / "clr-flt-range.glsl")
    DESCRIPTION = """
The SELECT RANGE GLSL (JOV) node applies a GLSL shader to select a specific range of colors within the input image. This node allows users to define the start and end points of the color range using RGB values, providing precise control over color selection. The GLSL shader used for this operation is loaded from the specified fragment file, enabling customizable color range selection for various image processing tasks.
"""

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
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw["start"] = parse_param(kw, Lexicon.START, EnumConvertType.VEC3, (0, 0, 0), 0, 1)
        kw.pop(Lexicon.START, None)
        kw["end"] = parse_param(kw, Lexicon.END, EnumConvertType.VEC3, (1, 1, 1), 0, 1)
        kw.pop(Lexicon.END, None)
        return super().run(**kw)

class GLSLColorGrayscale(GLSLBaseNode):
    NAME = "GRAYSCALE GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    FRAGMENT = str(JOV_GLSL / "clr" / "clr-grayscale.glsl")
    DEFAULT = (0.299, 0.587, 0.114)
    DESCRIPTION = """
The GRAYSCALE GLSL (JOV) node converts the input image to grayscale using a GLSL shader. This node applies a customizable grayscale conversion formula to each pixel of the input image, allowing users to specify the RGB weights for the conversion. The GLSL shader used for this operation is loaded from the specified fragment file, providing flexibility in grayscale conversion methods for different image processing requirements.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.RGB: ("VEC3", {"default": cls.DEFAULT, "step": 0.01, "precision": 4,
                                   "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw["conversion"] = parse_param(kw, Lexicon.RGB, EnumConvertType.VEC3, self.DEFAULT, 0, 1)
        kw.pop(Lexicon.RGB, None)
        return super().run(**kw)

class GLSLCreateNoise(GLSLBaseNode):
    NAME = "NOISE GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
The NOISE GLSL (JOV) node generates noise using GLSL shaders, providing various types of noise patterns for image processing applications. Users can select from different noise types, including Brownian, Gradient, Mosaic, Perlin 2D, Simplex 2D, and Value noise. The generated noise patterns can be customized further by specifying parameters such as seed and image dimensions. GLSL shaders corresponding to each noise type are loaded dynamically based on the user's selection, allowing for flexible and efficient noise generation in the image processing pipeline.
"""

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
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw[Lexicon.FRAGMENT] = []
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumNoiseType.VALUE.name)
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
        kw["seed"] = parse_param(kw, Lexicon.SEED, EnumConvertType.INT, 0)
        kw.pop(Lexicon.SEED, None)
        return super().run(**kw)

class GLSLCreatePattern(GLSLBaseNode):
    NAME = "PATTERN GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
The PATTERN GLSL (JOV) node generates patterns using GLSL shaders, providing a variety of pattern types for image processing tasks. Users can select from different pattern types, including the checkerboard pattern. The generated patterns can be customized further by specifying parameters such as tile size and image dimensions. GLSL shaders corresponding to each pattern type are loaded dynamically based on the user's selection, enabling flexible and efficient pattern generation in the image processing pipeline.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.TYPE: (EnumPatternType._member_names_, {"default": EnumPatternType.CHECKER.name}),
            Lexicon.TILE: ("VEC2", {"default": (1, 1), "step": 0.02, "precision": 6, "min": 1, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]})
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw["uTile"] = parse_param(kw, Lexicon.TILE, EnumConvertType.VEC2, (1, 1), 1)
        kw.pop(Lexicon.TILE)
        kw[Lexicon.FRAGMENT] = []
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumPatternType.CHECKER.name)
        kw.pop(Lexicon.TYPE, None)
        for t in typ:
            match EnumPatternType[t]:
                case EnumPatternType.CHECKER:
                    val = JOV_GLSL / "cre"/ "cre-pat-checker.glsl"
            kw[Lexicon.FRAGMENT].append(val)
        return super().run(**kw)

class GLSLCreatePolygon(GLSLBaseNode):
    NAME = "POLYGON GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    FRAGMENT = str(JOV_GLSL / "cre" / "cre-shp-polygon.glsl")
    DESCRIPTION = """
The POLYGON GLSL (JOV) node generates polygonal shapes using GLSL shaders. Users can specify the number of sides for the polygon and its radius, allowing for the creation of various polygonal shapes such as triangles, squares, pentagons, and more. The generated shapes can be further processed within the image processing pipeline. GLSL shaders corresponding to polygon generation are dynamically loaded, enabling efficient shape generation and manipulation.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.VALUE: ("INT", {"default": 3, "step": 1, "min": 3}),
            Lexicon.RADIUS: ("FLOAT", {"default": 1, "min": 0.01, "max": 4, "step": 0.01}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]})
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw["sides"] = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 3, 3)
        kw.pop(Lexicon.VALUE, None)
        val = parse_param(kw, Lexicon.RADIUS, EnumConvertType.FLOAT, 1, 1)
        kw["radius"] = [1. / v for v in val]
        kw.pop(Lexicon.RADIUS, None)
        return super().run(**kw)

class GLSLMap(GLSLBaseNode):
    NAME = "MAP GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
The MAP GLSL (JOV) node applies mapping transformations to input images using GLSL shaders. It offers various mapping types such as polar mapping, Mercator projection, and rectangular equal-area projection. Users can choose the desired mapping type and optionally flip the output image. GLSL shaders corresponding to different mapping transformations are dynamically loaded, enabling efficient image mapping operations within the image processing pipeline.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.TYPE: (EnumMappingType._member_names_, {"default": EnumMappingType.POLAR.name}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw[Lexicon.FRAGMENT] = []
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumMappingType.POLAR.name)
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
        kw["flip"] = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        kw.pop(Lexicon.FLIP, None)
        return super().run(**kw)

class GLSLTRSMirror(GLSLBaseNode):
    NAME = "MIRROR GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-mirror.glsl")
    DESCRIPTION = """
Applies a mirror transformation to an image using GLSL. Allows for setting the angle of mirroring and the pivot point.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize, "step": 0.01}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw["center"] = parse_param(kw, Lexicon.PIVOT, EnumConvertType.VEC2, (0.5, 0.5), 0, 1)
        kw.pop(Lexicon.PIVOT, None)
        kw["uZoom"] = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        kw["uZoom"] = [-a for a in kw["uZoom"]]
        kw.pop(Lexicon.ANGLE, None)
        return super().run(**kw)

class GLSLTRSRotate(GLSLBaseNode):
    NAME = "ROTATE GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-rotate.glsl")
    DESCRIPTION = """
Applies a rotation transformation to an image using GLSL. Allows for setting the angle of rotation and the pivot point.
"""
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize, "step": 0.01}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw["center"] = parse_param(kw, Lexicon.PIVOT, EnumConvertType.VEC2, (0.5, 0.5), 0, 1)
        kw.pop(Lexicon.PIVOT, None)
        kw["angle"] = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        kw["angle"] = [-a for a in kw["angle"]]
        kw.pop(Lexicon.ANGLE, None)
        return super().run(**kw)

class GLSLUtilTiler(GLSLBaseNode):
    NAME = "TILER GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-tiler.glsl")
    DESCRIPTION = """
Applies a tiling effect to an image using GLSL. Allows for setting the number of tiles in the x and y directions.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.TILE: ("VEC2", {"default": (1., 1., ), "min": 1, "step": 0.1, "precision": 4,
                                     "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw["uTile"] = parse_param(kw, Lexicon.TILE, EnumConvertType.VEC2, (1, 1), 1)
        kw.pop(Lexicon.TILE)
        return super().run(**kw)

class GLSLTRSKaleidoscope(GLSLBaseNode):
    NAME = "KALEIDOSCOPE GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    FRAGMENT = str(JOV_GLSL / "trs" / "trs-kaleidoscope.glsl")
    DESCRIPTION = """
Applies a kaleidoscope effect to an image using GLSL. Allows for adjusting various parameters such as segments, radius, zoom, offset, rotation, size, and skip to create intricate patterns.
"""

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
                                       "min": -sys.maxsize, "max": sys.maxsize,
                                       "tooltip":"spin the input texture for each segment"}),
            Lexicon.SIZE: ("FLOAT", {"default": 0.5, "step": 0.002, "precision": 6,
                                     "min": 0.00001, "max": 10}),
            Lexicon.SKIP: ("FLOAT", {"default": 0, "step": 0.002, "precision": 6,
                                     "min": -0.5, "max": 0.5}),
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw["segments"] = parse_param(kw, Lexicon.SEGMENT, EnumConvertType.FLOAT, 2, 2.5)
        kw.pop(Lexicon.SEGMENT)
        kw["radius"] = parse_param(kw, Lexicon.RADIUS, EnumConvertType.FLOAT, 1, 0.00001, 2)
        kw.pop(Lexicon.RADIUS)
        kw["regress"] = parse_param(kw, Lexicon.ZOOM, EnumConvertType.FLOAT, 1, 0, 2)
        kw.pop(Lexicon.ZOOM)
        kw["shift"] = parse_param(kw, Lexicon.OFFSET, EnumConvertType.VEC2, (0.5, 0.5), 0, 1)
        kw.pop(Lexicon.OFFSET)
        kw["spin"] = parse_param(kw, Lexicon.ROTATE, EnumConvertType.FLOAT, 0)
        kw.pop(Lexicon.ROTATE)
        kw["scale"] = parse_param(kw, Lexicon.SIZE, EnumConvertType.FLOAT, 0.5, 0.00001, 10)
        kw.pop(Lexicon.SIZE)
        kw["skip"] = parse_param(kw, Lexicon.SKIP, EnumConvertType.FLOAT, 0, 0.5, -0.5)
        kw.pop(Lexicon.SKIP)
        return super().run(**kw)

class GLSLVFX(GLSLBaseNode):
    NAME = "VFX GLSL (JOV)"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Applies various visual effects to an image using GLSL. The effects include bulge, chromatic aberration, cross-hatch, CRT, film grain, frosted glass, pixelation, sepia, and VHS. Each effect can be customized with parameters such as radius, strength, and center.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            "radius": ("FLOAT", {"default": 2., "min": 0.0001, "step": 0.01}),
            "strength": ("FLOAT", {"default": 1., "min": 0., "step": 0.01}),
            "center": ("VEC2", {"default": (0.5, 0.5,), "min": 0, "max": 1, "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.TYPE: (EnumVFXType._member_names_, {"default": EnumVFXType.BULGE.name})
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> List[torch.Tensor]:
        kw[Lexicon.FRAGMENT] = []
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType.STRING, EnumVFXType.BULGE.name)
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
