"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import re
import math
import time
from typing import Any, Optional

import cv2
import torch
import numpy as np
from PIL import Image

from Jovimetrix import pil2tensor, pil2mask, pil2cv, cv2pil, cv2tensor, cv2mask, tensor2cv, \
    deep_merge_dict, zip_longest_fill, \
    JOVImageBaseNode, JOVImageInOutBaseNode, Logger, \
    IT_PIXELS, IT_COLOR, IT_WH, IT_SCALE, IT_ROT, IT_INVERT, \
    IT_WHMODE, IT_REQUIRED, MIN_HEIGHT, MIN_WIDTH

from Jovimetrix.sup.comp import EnumScaleMode, geo_scalefit, shape_ellipse, shape_polygon, shape_quad, light_invert, \
    EnumInterpolation, IT_SAMPLE

# =============================================================================

class ConstantNode(JOVImageBaseNode):
    NAME = "🟪 Constant (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_WH, IT_COLOR)

    def run(self, width: int, height: int, R: float, G: float, B: float) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.new("RGB", (width, height), (int(R * 255.), int(G * 255.), int(B * 255.)) )
        return (pil2tensor(image), pil2tensor(image.convert("L")),)

class ShapeNode(JOVImageBaseNode):
    NAME = "✨ Shape Generator (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {
                "shape": (["CIRCLE", "SQUARE", "ELLIPSE", "RECTANGLE", "POLYGON"], {"default": "SQUARE"}),
                "sides": ("INT", {"default": 3, "min": 3, "max": 100, "step": 1}),
            },
        }
        return deep_merge_dict(d, IT_WH, IT_COLOR, IT_ROT, IT_SCALE, IT_INVERT)

    def run(self, shape: str, sides: int, width: int, height: int, R: float, G: float, B: float,
            angle: float, sizeX: float, sizeY: float, invert: float) -> tuple[torch.Tensor, torch.Tensor]:

        image = None
        fill = (int(R * 255.),
                int(G * 255.),
                int(B * 255.),)

        match shape:
            case 'SQUARE':
                image = shape_quad(width, height, sizeX, sizeX, fill=fill)

            case 'ELLIPSE':
                image = shape_ellipse(width, height, sizeX, sizeY, fill=fill)

            case 'RECTANGLE':
                image = shape_quad(width, height, sizeX, sizeY, fill=fill)

            case 'POLYGON':
                image = shape_polygon(width, height, sizeX, sides, fill=fill)

            case _:
                image = shape_ellipse(width, height, sizeX, sizeX, fill=fill)

        image = image.rotate(-angle)
        if invert > 0.:
            image = pil2cv(image)
            image = light_invert(image, invert)
            image = cv2pil(image)

        return (pil2tensor(image), pil2tensor(image.convert("L")), )

class PixelShaderNode(JOVImageInOutBaseNode):
    NAME = "🔆 Pixel Shader (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "optional": {
                "R": ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
                "G": ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
                "B": ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
            },
        }
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_WHMODE, IT_SAMPLE)

    @staticmethod
    def shader(image:np.ndarray,
               R: str, G: str, B: str,
               width: int, height: int,
               chunkX: int=64, chunkY:int=64) -> np.ndarray:

        from ast import literal_eval

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
                    "$u": x / width,
                    "$v": y / height,
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

    def run(self,
            pixels: Optional[list[torch.tensor]]=None,
            R: Optional[list[str]]=None,
            G: Optional[list[str]]=None,
            B: Optional[list[str]]=None,
            width: Optional[list[int]]=None,
            height: Optional[list[int]]=None,
            mode: Optional[list[str]]=None,
            resample: Optional[list[str]]=None) -> tuple[torch.Tensor, torch.Tensor]:

        run = time.perf_counter()
        pixels = pixels or [None]
        R = R or [None]
        G = G or [None]
        B = B or [None]
        width = width or [None]
        height = height or [None]
        mode = mode or [None]
        resample = resample or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, R, G, B, width, height, mode, resample):
            image, r, g, b, w, h, m, rs = data

            r = r if r is not None else ""
            g = g if g is not None else ""
            b = b if b is not None else ""
            w = w if w is not None else MIN_WIDTH
            h = h if h is not None else MIN_HEIGHT
            m = m if m is not None else EnumScaleMode.FIT
            m = EnumScaleMode.FIT if m == EnumScaleMode.NONE else m

            # fix the image first -- must at least match for px, py indexes
            image = geo_scalefit(image, w, h, m, rs)

            if image is None:
                image = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                image = tensor2cv(image)
                if image.shape[0] != h or image.shape[1] != w:
                    s = EnumInterpolation.LANCZOS4.value
                    rs = EnumInterpolation[rs].value if rs is not None else s
                    image = cv2.resize(image, (w, h), interpolation=rs)

            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4

            image = PixelShaderNode.shader(image, r, g, b, w, h)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

        Logger.info(self.NAME, {time.perf_counter() - run:.5})
        return (
            torch.stack(images),
            torch.stack(masks)
        )

class GLSLNode(JOVImageBaseNode):
    NAME = "🍩 GLSL (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )
    POST = True

    @classmethod
    def INPUT_TYPES(s) -> dict[str, dict]:
        d =  {
            "required": {
                "vertex": ("STRING", {"default":
"""attribute vec4 a_position;
void main() {
    gl_Position = a_position;
}
""", "multiline": True}),

                "fragment": ("STRING", {"default":
"""precision mediump float;
void main() {
    gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0); // Blue color
}
""", "multiline": True}),
}}
        return deep_merge_dict(d, IT_WH)

    @classmethod
    def IS_CHANGED(cls, *arg, **kw) -> Any:
        return float("nan")

    def run(self, vertex: str, fragment: str, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.new(mode="RGB", size=(200, 200))
        return (pil2tensor(image), pil2mask(image))

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass