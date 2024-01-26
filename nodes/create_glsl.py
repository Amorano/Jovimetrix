"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation - GLSL
"""

import torch
from loguru import logger

import comfy
from server import PromptServer

from Jovimetrix import IT_WH, JOV_GLSL, ComfyAPIMessage, JOVBaseNode, \
    ROOT, IT_PIXELS, IT_REQUIRED, MIN_IMAGE_SIZE, TimedOutException

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import EnumTupleType, deep_merge_dict, parse_tuple
from Jovimetrix.sup.image import pil2tensor, tensor2pil
from Jovimetrix.sup.shader import GLSL, CompileException

JOV_CONFIG_GLSL = ROOT / 'glsl'

DEFAULT_FRAGMENT = """void main() {
    vec4 texColor = texture(iChannel0, iUV);
    vec4 color = vec4(iUV, abs(sin(iTime)), 1.0);
    fragColor = vec4((texColor.xyz + color.xyz) / 2.0, 1.0);
}"""

DEFAULT_FRAGMENT = """void main() {
    vec4 texColor = texture(iChannel0, iUV);
    vec4 color = vec4(iUV, abs(sin(iTime)), 1.0);
    fragColor = vec4((texColor.xyz + color.xyz) / 2.0, 1.0);
}"""

# =============================================================================

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
        param = kw.get(Lexicon.PARAM, {})
        width, height = parse_tuple(Lexicon.WH, kw, default=(self.WIDTH, self.HEIGHT,), clip_min=1)[0]
        if self.__fragment != fragment or self.__glsl is None:
            try:
                self.__glsl = GLSL(fragment, width, height, param)
            except CompileException as e:
                PromptServer.instance.send_sync("jovi-glsl-error", {"id": id, "e": str(e)})
                logger.error(e)
                return (self.__last_good, )
            self.__fragment = fragment

        self.__glsl.width = width
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
            img = self.__glsl.render(texture1, param)
            frames.append(pil2tensor(img))
            pbar.update_absolute(idx)

        runtime = self.__glsl.runtime if not reset else 0
        PromptServer.instance.send_sync("jovi-glsl-time", {"id": id, "t": runtime})

        self.__last_good = frames
        return (self.__last_good, )

class GLSLBaseNode(JOVBaseNode):
    CATEGORY = "JOVIMETRIX GLSL"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    FRAGMENT = ".glsl"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS)

    def run(self, param:dict=None, **kw) -> list[torch.Tensor]:
        if (texture1 := kw.get(Lexicon.PIXEL, None)) is not None:
            texture1 = tensor2pil(texture1)

        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        width, height = wihi
        img = GLSL.instant(str(self.FRAGMENT), texture1=texture1, width=width, height=height, param=param)
        return (pil2tensor(img),)

class GLSLGrayscale(GLSLBaseNode):
    NAME = "GRAYSCALE GLSL (JOV)"
    CATEGORY = "JOVIMETRIX GLSL/COLOR"
    FRAGMENT = JOV_GLSL / "color-grayscale.glsl"
    DEFAULT = (0.299, 0.587, 0.114)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        e = {"optional": {
            Lexicon.RGB: ("VEC3", {"default": cls.DEFAULT, "step": 0.01, "min": 0, "max": 1, "precision": 4, "round": 0.00001, "label": [Lexicon.R, Lexicon.G, Lexicon.B]}),
        }}
        return deep_merge_dict(d, e)

    def run(self, **kw) -> list[torch.Tensor]:
        rgb = kw.pop(Lexicon.RGB, self.DEFAULT)
        param = {"conversion": rgb}
        return super().run(param, **kw)

class GLSLNoise1D(GLSLBaseNode):
    NAME = "NOISE 1D GLSL (JOV)"
    CATEGORY = "JOVIMETRIX GLSL/NOISE"
    FRAGMENT = JOV_GLSL / "noise-1D.glsl"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        e = {"optional": {
            Lexicon.VALUE: ("INT", {"default": 0, "step": 1}),
        }}
        return deep_merge_dict(IT_REQUIRED, e, IT_WH)

    def run(self, **kw) -> list[torch.Tensor]:
        seed = kw.pop(Lexicon.VALUE, 0)
        param = {"seed": seed}
        return super().run(param, **kw)

class GLSLNoise2DSimplex(GLSLBaseNode):
    NAME = "NOISE 2D SIMPLEX GLSL (JOV)"
    CATEGORY = "JOVIMETRIX GLSL/NOISE"
    FRAGMENT = JOV_GLSL / "noise-2D-simplex.glsl"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        e = {"optional": {
            Lexicon.VALUE: ("INT", {"default": 0, "step": 1}),
        }}
        return deep_merge_dict(IT_REQUIRED, e, IT_WH)

    def run(self, **kw) -> list[torch.Tensor]:
        seed = kw.pop(Lexicon.VALUE, 0)
        param = {"seed": seed}
        return super().run(param, **kw)

class GLSLPolygon(GLSLBaseNode):
    NAME = "POLYGON GLSL (JOV)"
    CATEGORY = "JOVIMETRIX GLSL/CREATE"
    FRAGMENT = JOV_GLSL / "shape-polygon.glsl"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        e = {"optional": {
            Lexicon.VALUE: ("INT", {"default": 3, "step": 1, "min": 3}),
            Lexicon.RADIUS: ("FLOAT", {"default": 1, "min": 0.01, "max": 4, "step": 0.01}),
        }}
        return deep_merge_dict(IT_REQUIRED, e, IT_WH)

    def run(self, **kw) -> list[torch.Tensor]:
        sides = kw.pop(Lexicon.VALUE, 3)
        radius = kw.pop(Lexicon.RADIUS, 1)
        param = {"sides": sides, "radius": 1. / radius}
        return super().run(param, **kw)

class GLSLTransRotate(GLSLBaseNode):
    NAME = "ROTATE GLSL (JOV)"
    CATEGORY = "JOVIMETRIX GLSL/ADJUST"
    FRAGMENT = JOV_GLSL / "trans-rotate.glsl"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        e = {"optional": {
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "step": 0.01}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "max": 1, "min": 0, "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return deep_merge_dict(d, e)

    def run(self, **kw) -> list[torch.Tensor]:
        angle = kw.pop(Lexicon.ANGLE, 0)
        center = parse_tuple(Lexicon.PIVOT, kw, typ=EnumTupleType.FLOAT, default=(0.5, 0.5,), clip_min=0, clip_max=1)[0]
        param = {"angle": -angle, "center": center}
        return super().run(param, **kw)

class GLSLUtilTiler(GLSLBaseNode):
    NAME = "TILER GLSL (JOV)"
    CATEGORY = "JOVIMETRIX GLSL/UTIL"
    FRAGMENT = JOV_GLSL / "util-tiler.glsl"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        e = {"optional": {
            "uTime": ("FLOAT", {"default": 0, "step": 0.01}),
            "uTile": ("VEC2", {"default": (1., 1., ), "min": 1., "step": 0.01, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
        }}
        return deep_merge_dict(d, e)

    def run(self, **kw) -> list[torch.Tensor]:
        uTime = kw.pop("uTime", 0.)
        uTile = parse_tuple("uTile", kw, typ=EnumTupleType.FLOAT, default=(1., 1.,), clip_min=1)[0]
        param = {"uTime": uTime, "uTile": uTile}
        return super().run(param, **kw)
