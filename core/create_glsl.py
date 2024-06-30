"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation - GLSL
"""

from typing import List

import cv2
import torch

from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import comfy_message, parse_reset, WILDCARD, ROOT
from Jovimetrix.sup.lexicon import JOVImageNode, Lexicon
from Jovimetrix.sup.util import parse_dynamic, parse_param, zip_longest_fill, EnumConvertType
from Jovimetrix.sup.image import  cv2tensor_full, pil2cv, tensor2pil, MIN_IMAGE_SIZE
from Jovimetrix.sup.shader import GLSLShader, MAX_WIDTH, MAX_HEIGHT, CompileException

# =============================================================================

JOV_CATEGORY = "CREATE"
JOV_CONFIG_GLSL = ROOT / 'glsl'
# =============================================================================

'''
class GLSLNode(JOVImageNode):
    NAME = "GLSL (JOV) 🍩"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 1
    DESCRIPTION = """
The GLSL Node executes custom GLSL (OpenGL Shading Language) fragment shaders to generate images or apply effects. GLSL is a high-level shading language used for graphics programming, particularly in the context of rendering images or animations. This node allows for real-time rendering of shader effects, providing flexibility and creative control over image processing pipelines. It takes advantage of GPU acceleration for efficient computation, enabling the rapid generation of complex visual effects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (WILDCARD, {}),
                Lexicon.TIME: ("FLOAT", {"default": 0, "step": 0.0001, "min": 0, "precision": 6}),
                Lexicon.BATCH: ("VEC2", {"default": (1, 30), "step": 1, "label": ["COUNT", "FPS"], "tooltip": "Number of frames wanted and the FPS"}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
                Lexicon.RESET: ("BOOLEAN", {"default": False}),
                Lexicon.WH: ("VEC2", {"default": (512, 512), "min":MIN_IMAGE_SIZE, "step": 1,}),
                Lexicon.FRAGMENT: ("STRING", {"multiline": True, "default": cls.FRAGMENT, "dynamicPrompts": False}),
                Lexicon.PARAM: ("STRING", {"default": {}})
            }

        })
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
        fragment = parse_param(kw, Lexicon.FRAGMENT, EnumConvertType.STRING, self.PROG_FRAGMENT)
        param = parse_param(kw, Lexicon.PARAM, EnumConvertType.DICT, {})
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        pA = parse_param(kw, Lexicon.GLSL, EnumConvertType.IMAGE, None)
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
                image = self.__glsl.render(pA, param)
                image = pil2cv(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                images.append(cv2tensor_full(image))
            runtime = self.__glsl.runtime if not reset else 0
            comfy_message(ident, "jovi-glsl-time", {"id": ident, "t": runtime})

            self.__last_good = images
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]
'''

class GLSLNode(JOVImageNode):
    NAME = "GLSL (JOV) 🍩"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 1
    DESCRIPTION = """
The GLSL Node executes custom GLSL (OpenGL Shading Language) fragment shaders to generate images or apply effects. GLSL is a high-level shading language used for graphics programming, particularly in the context of rendering images or animations. This node allows for real-time rendering of shader effects, providing flexibility and creative control over image processing pipelines. It takes advantage of GPU acceleration for efficient computation, enabling the rapid generation of complex visual effects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.GLSL: (WILDCARD, {}),
                Lexicon.TIME: ("FLOAT", {"default": 0, "step": 0.0001, "min": 0, "precision": 6}),
                Lexicon.BATCH: ("INT", {"default": 1, "step": 1, "min": 1, "max": 262144}),
                Lexicon.FPS: ("INT", {"default": 30, "step": 1, "min": 1, "max": 120}),
                Lexicon.WH: ("VEC2", {"default": (512, 512), "min": MIN_IMAGE_SIZE, "step": 1,}),
                Lexicon.WAIT: ("BOOLEAN", {"default": False}),
                Lexicon.RESET: ("BOOLEAN", {"default": False}),
                Lexicon.FRAGMENT: ("STRING", {"default": GLSLShader.PROG_FRAGMENT, "multiline": True, "dynamicPrompts": False})
            }
        })
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__glsl = GLSLShader()
        self.__fragment = ""

    def run(self, **kw) -> tuple[torch.Tensor]:
        inputs = parse_dynamic(kw, Lexicon.GLSL, EnumConvertType.ANY, None)
        delta = parse_param(kw, Lexicon.TIME, EnumConvertType.FLOAT, 0)
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 1, 1, 262144)
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 1, 1, 120)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        fragment = parse_param(kw, Lexicon.FRAGMENT, EnumConvertType.STRING, GLSLShader.PROG_FRAGMENT)
        wait = parse_param(kw, Lexicon.WAIT, EnumConvertType.BOOLEAN, False)
        reset = parse_param(kw, Lexicon.RESET, EnumConvertType.BOOLEAN, False)
        images = []
        params = list(zip_longest_fill(inputs, delta, batch, fps, wihi, fragment, wait, reset))
        pbar = ProgressBar(len(params))
        for idx, (inputs, delta, batch, fps, wihi, fragment, wait, reset) in enumerate(params):
            self.__glsl.fragment = fragment
            self.__glsl.size = wihi
            self.__glsl.fps = fps
            self.__glsl.runtime = delta
            # update_var(shader, width, height, frame * (1.0 / fps), (1.0 / fps), fps, frame)
            #if channel_0 is not None: self.update_texture(textures[0], channel_0)
            #if channel_1 is not None: self.update_texture(textures[1], channel_1)
            #if channel_2 is not None: self.update_texture(textures[2], channel_2)
            #if channel_3 is not None: self.update_texture(textures[3], channel_3)
            #self.__glsl.texture_bind(shader, textures)
            image = self.__glsl.render()
            images.append(image)
            pbar.update_absolute(idx)

        self.__glsl.render_resources_cleanup(ctx)
        return (torch.cat(images, dim=0),)

