"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import re
import math
import time
from enum import Enum
from typing import Any

import cv2
import torch
import numpy as np
from PIL import Image

from Jovimetrix import pil2tensor, pil2mask, pil2cv, cv2pil, cv2tensor, cv2mask, \
    tensor2cv, deep_merge_dict, zip_longest_fill, \
    JOVImageBaseNode, JOVImageInOutBaseNode, Logger, Lexicon, \
    TYPE_PIXEL, IT_PIXELS, IT_RGB, IT_WH, IT_SCALE, IT_ROT, IT_INVERT, \
    IT_WHMODE, IT_REQUIRED, MIN_HEIGHT, MIN_WIDTH

from Jovimetrix.sup.comp import EnumScaleMode, geo_scalefit, shape_ellipse, \
    shape_polygon, shape_quad, light_invert, \
    EnumInterpolation, IT_SAMPLE

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
        return deep_merge_dict(IT_REQUIRED, IT_RGB, IT_WH)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        color = kw.get(Lexicon.RGB, (255, 255, 255))
        width = kw.get(Lexicon.WIDTH, 0)
        height = kw.get(Lexicon.HEIGHT, 0)
        image = Image.new("RGB", (width, height), color)
        return (pil2tensor(image), pil2tensor(image.convert("L")),)

class ShapeNode(JOVImageBaseNode):
    NAME = "SHAPE GENERATOR (JOV) ✨"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {
                Lexicon.SHAPE: (EnumShapes._member_names_, {"default": EnumShapes.CIRCLE.name}),
                Lexicon.SIDES: ("INT", {"default": 3, "min": 3, "max": 100, "step": 1}),
            },
        }
        return deep_merge_dict(d, IT_WH, IT_RGB, IT_ROT, IT_SCALE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        i = kw.get(Lexicon.INVERT, 0)
        shape = kw.get(Lexicon.SHAPE, EnumShapes.CIRCLE)
        sides = kw.get(Lexicon.SIDES, 3)
        angle = kw.get(Lexicon.ANGLE, 0)
        sizeX = kw.get(Lexicon.SIZE_X, 1)
        sizeY = kw.get(Lexicon.SIZE_Y, 1)
        width = kw.get(Lexicon.WIDTH, 0)
        height = kw.get(Lexicon.HEIGHT, 0)
        color = kw.get(Lexicon.RGB, (255, 255, 255))
        R, G, B = color
        image = None
        fill = (int(R * 255.), int(G * 255.), int(B * 255.),)

        match shape:
            case EnumShapes.SQUARE:
                image = shape_quad(width, height, sizeX, sizeX, fill=fill)

            case EnumShapes.ELLIPSE:
                image = shape_ellipse(width, height, sizeX, sizeY, fill=fill)

            case EnumShapes.RECTANGLE:
                image = shape_quad(width, height, sizeX, sizeY, fill=fill)

            case EnumShapes.POLYGON:
                image = shape_polygon(width, height, sizeX, sides, fill=fill)

            case EnumShapes.CIRCLE:
                image = shape_ellipse(width, height, sizeX, sizeX, fill=fill)

        image = image.rotate(-angle)
        if (i or 0) > 0.:
            image = pil2cv(image)
            image = light_invert(image, i)
            image = cv2pil(image)

        return (pil2tensor(image), pil2tensor(image.convert("L")), )

class PixelShaderNode(JOVImageInOutBaseNode):
    NAME = "PIXEL SHADER (JOV) 🔆"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "optional": {
                Lexicon.R: ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
                Lexicon.G: ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
                Lexicon.B: ("STRING", {"multiline": True, "default": "1 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5))"}),
            },
        }
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_WHMODE, IT_SAMPLE)

    @staticmethod
    def shader(image: TYPE_PIXEL, R: str, G: str, B: str, chunkX: int=64, chunkY:int=64, **kw) -> np.ndarray:

        from ast import literal_eval
        width = kw.get(Lexicon.WIDTH, 0)
        height = kw.get(Lexicon.HEIGHT, 0)
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
        width = kw.get(Lexicon.WIDTH, [None])
        height = kw.get(Lexicon.HEIGHT, [None])
        mode = kw.get(Lexicon.MODE, [None])
        resample = kw.get(Lexicon.RESAMPLE, [None])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, R, G, B, width, height, mode, resample):
            image, r, g, b, w, h, m, rs = data

            r = r if r else ""
            g = g if g else ""
            b = b if b else ""
            w = w if w else MIN_WIDTH
            h = h if h else MIN_HEIGHT
            m = m if m else EnumScaleMode.FIT
            m = EnumScaleMode.FIT if m == EnumScaleMode.NONE else m

            # fix the image first -- must at least match for px, py indexes
            image = geo_scalefit(image, w, h, m, rs)

            if image is None:
                image = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                image = tensor2cv(image)
                if image.shape[0] != h or image.shape[1] != w:
                    s = EnumInterpolation.LANCZOS4
                    rs = EnumInterpolation[rs] if rs else s
                    image = cv2.resize(image, (w, h), interpolation=rs)

            rs = EnumInterpolation[rs] if rs else EnumInterpolation.LANCZOS4
            image = PixelShaderNode.shader(image, r, g, b, w, h)
            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

        Logger.info(self.NAME, {time.perf_counter() - t:.5})
        return (
            torch.stack(images),
            torch.stack(masks)
        )

class GLSLNode(JOVImageBaseNode):
    NAME = "GLSL (JOV) 🍩"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict]:
        d =  {
            "required": {
                Lexicon.VERTEX: ("STRING", {"default":
"""attribute vec4 a_position;
void main() {
    gl_Position = a_position;
}
""", "multiline": True}),

                Lexicon.FRAGMENT: ("STRING", {"default":
"""precision mediump float;
void main() {
    gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0); // Blue color
}
""", "multiline": True}),
}}
        return deep_merge_dict(d, IT_WH)

    @classmethod
    def IS_CHANGED(cls, *arg, **kw) -> float:
        return float("nan")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        width = kw.get(Lexicon.WIDTH, 0)
        height = kw.get(Lexicon.HEIGHT, 0)
        vertex = kw.get(Lexicon.VERTEX, '')
        fragment = kw.get(Lexicon.FRAGMENT, '')
        image = Image.new(mode="RGB", size=(width, height))
        return (pil2tensor(image), pil2mask(image))

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass