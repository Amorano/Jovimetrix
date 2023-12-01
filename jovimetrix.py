"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix

                    Copyright 2023 Alexander Morano (Joviex)
"""

import os
import gc
import re
import json
import time
from typing import Any

import cv2
import torch
import numpy as np
from PIL import Image

import comfy

from .sup import util
from .sup import stream
from .sup import comp
from .sup.anim import ease, EaseOP
from .sup.util import loginfo, logwarn, logerr, logdebug

# =============================================================================

JOV_MAX_DELAY = 60.
try: JOV_MAX_DELAY = float(os.getenv("JOV_MAX_DELAY", 60.))
except: pass

# =============================================================================
# === CORE NODES ===
# =============================================================================

class JovimetrixBaseNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return IT_REQUIRED

    NAME = "Jovimetrix"
    DESCRIPTION = "A Jovimetrix Node"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ()
    OUTPUT_NODE = False
    INPUT_IS_LIST = False
    FUNCTION = "run"

class JovimetrixImageBaseNode(JovimetrixBaseNode):
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("🖼️", "😷",)
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, True, )

class JovimetrixImageInOutBaseNode(JovimetrixBaseNode):
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("🖼️", "😷",)
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, True, )

# wildcard trick is 100% stolen from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

WILDCARD = AnyType("*")

# =============================================================================
# === GLOBAL SUPPORTS ===
# =============================================================================

def deep_merge_dict(*dicts: dict) -> dict:
    """
    Deep merge multiple dictionaries recursively.
    """
    def _deep_merge(d1, d2):
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            return d2

        merged_dict = d1.copy()

        for key in d2:
            if key in merged_dict:
                if isinstance(merged_dict[key], dict) and isinstance(d2[key], dict):
                    merged_dict[key] = _deep_merge(merged_dict[key], d2[key])
                elif isinstance(merged_dict[key], list) and isinstance(d2[key], list):
                    merged_dict[key].extend(d2[key])
                else:
                    merged_dict[key] = d2[key]
            else:
                merged_dict[key] = d2[key]
        return merged_dict

    merged = {}
    for d in dicts:
        merged = _deep_merge(merged, d)
    return merged

IT_REQUIRED = {
    "required": {}
}

IT_IMAGE = {
    "required": {
        "image": ("IMAGE", ),
    }}

IT_PIXELS = {
    "optional": {
        "pixels": (WILDCARD, {}),
    }}

IT_PIXEL2 = {
    "required": {
        "pixelA": (WILDCARD, {}),
    },
    "optional": {
        "pixelB": (WILDCARD, {}),
    }}

IT_WH = {
    "optional": {
        "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 1, "display": "number"}),
        "height": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 1, "display": "number"}),
    }}

IT_WHMODE = {
    "optional": {
        "mode": (["NONE", "FIT", "CROP", "ASPECT"], {"default": "NONE"}),
        "resample": ([e.name for e in Image.Resampling], {"default": Image.Resampling.LANCZOS.name}),
    }}

IT_TRANS = {
    "optional": {
        "offsetX": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01, "display": "number"}),
        "offsetY": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01, "display": "number"}),
    }}

IT_ROT = {
    "optional": {
        "angle": ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 1, "display": "number"}),
    }}

IT_SCALE = {
    "optional": {
        "sizeX": ("FLOAT", {"default": 1, "min": 0.01, "max": 2., "step": 0.01, "display": "number"}),
        "sizeY": ("FLOAT", {"default": 1, "min": 0.01, "max": 2., "step": 0.01, "display": "number"}),
    }}

IT_SAMPLE = {
    "optional": {
        "resample": ([e.name for e in Image.Resampling], {"default": Image.Resampling.LANCZOS.name}),
    }}

IT_TILE = {
    "optional": {
        "tileX": ("INT", {"default": 1, "min": 0, "step": 1, "display": "number"}),
        "tileY": ("INT", {"default": 1, "min": 0, "step": 1, "display": "number"}),
    }}

IT_EDGE = {
    "optional": {
        "edge": (["CLIP", "WRAP", "WRAPX", "WRAPY"], {"default": "CLIP"}),
    }}

IT_INVERT = {
    "optional": {
        "invert": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
    }}

IT_COLOR = {
    "optional": {
        "R": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": "number"}),
        "G": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": "number"}),
        "B": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": "number"}),
    }}

IT_ORIENT = {
    "optional": {
        "orient": (["NORMAL", "FLIPX", "FLIPY", "FLIPXY"], {"default": "NORMAL"}),
    }}

IT_CAM = {
    "optional": {
        "zoom": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0}),
    }}

IT_TRS = deep_merge_dict(IT_TRANS, IT_ROT, IT_SCALE)

IT_WHMODEI = deep_merge_dict(IT_WH, IT_WHMODE, IT_INVERT, IT_SAMPLE)

# =============================================================================
# === CREATION NODES ===
# =============================================================================

class ConstantNode(JovimetrixImageBaseNode):
    NAME = "🟪 Constant (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_WH, IT_COLOR)

    def run(self, width: int, height: int, R: float, G: float, B: float) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.new("RGB", (width, height), (int(R * 255.), int(G * 255.), int(B * 255.)) )
        return (comp.pil2tensor(image), comp.pil2tensor(image.convert("L")),)

class ShapeNode(JovimetrixImageBaseNode):
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
                image = comp.sh_quad(width, height, sizeX, sizeX, fill=fill)

            case 'ELLIPSE':
                image = comp.sh_ellipse(width, height, sizeX, sizeY, fill=fill)

            case 'RECTANGLE':
                image = comp.sh_quad(width, height, sizeX, sizeY, fill=fill)

            case 'POLYGON':
                image = comp.sh_polygon(width, height, sizeX, sides, fill=fill)

            case _:
                image = comp.sh_ellipse(width, height, sizeX, sizeX, fill=fill)

        image = image.rotate(-angle)
        if invert > 0.:
            image = comp.pil2cv(image)
            image = comp.light_invert(image, invert)
            image = comp.cv2pil(image)

        return (comp.pil2tensor(image), comp.pil2tensor(image.convert("L")), )

class PixelShaderNode(JovimetrixImageInOutBaseNode):
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
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_WH, IT_SAMPLE)

    @staticmethod
    def shader(image: np.ndarray, R: str, G: str, B: str, width: int, height: int,
               chunkX: int=64, chunkY:int=64) -> np.ndarray:
        import math
        from ast import literal_eval

        out = np.zeros((height, width, 3), dtype=float)
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
                                logerr(f'eval failed {str(e)}\n{parseR}\n{parseG}\n{parseB}')

        return np.clip(out, 0, 255).astype(np.uint8)

    def run(self, R: list[str], G: list[str],
            B: list[str], width: list[int], height: list[int],
            resample: list[str], **kw) -> tuple[torch.Tensor, torch.Tensor]:

        run = time.time()
        pixels: list[torch.tensor] = kw.get('pixels',
                                            (torch.zeros((height[0], width[0], 3), dtype=torch.uint8),))

        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = comp.tensor2cv(image)
            rs = resample[min(idx, len(resample)-1)]
            rs = Image.Resampling[rs]
            w = width[min(idx, len(width)-1)]
            h = height[min(idx, len(height)-1)]
            image = cv2.resize(image, (w, h), interpolation=rs)
            _r, _g, _b = R[min(idx, len(R)-1)], G[min(idx, len(G)-1)], B[min(idx, len(B)-1)]

            image = PixelShaderNode.shader(image, _r, _g, _b, w, h)
            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        logdebug(f"[{self.NAME}] ({time.time() - run:.5})")
        return (
            torch.stack(images),
            torch.stack(masks)
        )

class GLSLNode(JovimetrixImageBaseNode):
    NAME = "🍩 GLSL (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (False, False, )
    POST = True

    @classmethod
    def INPUT_TYPES(s) -> dict[str, dict]:
        return {
            "required": {
                "vertex": ("STRING", {"default": """
                                #version 330

                                in vec2 in_vert;
                                void main() {
                                    gl_Position = vec4(in_vert, 0.0, 1.0);
                                    }
                                """, "multiline": True}),
                "fragment": ("STRING", {"default": """
                                #version 330

                                out vec4 fragColor;
                                void main() {
                                    fragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red color
                                }
                                """, "multiline": True}),
            }}

    def run(self, vertex: str, fragment: str) -> tuple[torch.Tensor, torch.Tensor]:
        import moderngl

        # @TODO: GET ACTUAL LITEGRAPH CONTEXT?
        ctx = moderngl.create_standalone_context(share=True)

        prog = ctx.program(vertex_shader=vertex, fragment_shader=fragment)

        # Create a simple quad
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        vbo = ctx.buffer(vertices)

        # Create a vertex array
        vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

        # Render the quad
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((512, 512), 3)])
        fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)

        # Read the pixel data
        data = np.frombuffer(fbo.read(components=3, dtype='f1'), dtype=np.float32)
        data = np.nan_to_num(data * 255., nan=0.)
        data = np.clip(data, 0, 255).astype(np.uint8)
        image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
        return (comp.pil2tensor(image), comp.pil2mask(image))

# =============================================================================
# === TRANSFORM NODES ===
# =============================================================================

class TransformNode(JovimetrixImageInOutBaseNode):
    NAME = "🌱 Transform (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Translate, Rotate, Scale, Tile and Invert an input. CROP or WRAP the edges."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_TRS, IT_EDGE, IT_WH, IT_WHMODE)

    def run(self, pixels: list[torch.tensor], offsetX: list[float],
            offsetY: list[float], angle: list[float], sizeX: list[float],
            sizeY: list[float], edge: list[str], width: list[int], height: list[int],
            mode: list[str], resample: list[str]) -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = comp.tensor2cv(image)
            rs = resample[min(idx, len(resample)-1)]
            rs = Image.Resampling[rs]
            image = comp.geo_transform(image,
                                   offsetX[min(idx, len(offsetX)-1)],
                                   offsetY[min(idx, len(offsetY)-1)],
                                   angle[min(idx, len(angle)-1)],
                                   sizeX[min(idx, len(sizeX)-1)],
                                   sizeY[min(idx, len(sizeY)-1)],
                                   edge[min(idx, len(edge)-1)],
                                   width[min(idx, len(width)-1)],
                                   height[min(idx, len(height)-1)],
                                   mode[min(idx, len(mode)-1)],
                                   rs,
                                )
            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class TileNode(JovimetrixImageInOutBaseNode):
    NAME = "🔳 Tile (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Tile an Image with optional crop to original image size."
    SORT = 5

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_TILE, IT_WH, IT_WHMODE)

    def run(self, pixels: list[torch.tensor], tileX: list[float], tileY: list[float],
            width: list[int], height: list[int], mode: list[str],
            resample: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = comp.tensor2cv(image)
            image = comp.geo_edge_wrap(image,
                                  tileX[min(idx, len(tileX)-1)],
                                  tileY[min(idx, len(tileY)-1)])

            rs = resample[min(idx, len(resample)-1)]
            rs = Image.Resampling[rs]
            image = comp.geo_scalefit(image,
                                  width[min(idx, len(width)-1)],
                                  height[min(idx, len(height)-1)],
                                  mode[min(idx, len(mode)-1)],
                                  rs)

            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class MirrorNode(JovimetrixImageInOutBaseNode):
    NAME = "🔰 Mirror (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Flip an input across the X axis, the Y Axis or both, with independent centers."
    SORT = 25

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "mode": (["X", "Y", "XY", "YX"], {"default": "X"}),
            },
        }
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self, pixels: list[torch.tensor], x: list[float], y: list[float],
            mode: list[str], invert: list[float]) -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = comp.tensor2cv(image)

            m = mode[min(idx, len(mode)-1)]
            i = invert[min(idx, len(invert)-1)]
            if 'X' in m:
                image = comp.geo_mirror(image, x, 1, invert=i)

            if 'Y' in m:
                image = comp.geo_mirror(image, y, 0, invert=i)

            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class MergeNode(JovimetrixImageInOutBaseNode):
    NAME = "➕ Merge (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Union multiple latents horizontal, vertical or in a grid."
    SORT = 15

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "axis": (["HORIZONTAL", "VERTICAL", "GRID"], {"default": "GRID"}),
            },
            "optional": {
                "stride": ("INT", {"min": 1, "step": 1, "default": 5}),
                "flip": ("BOOLEAN", {"default": False}),
            },
        }
        return deep_merge_dict(IT_PIXEL2, d, IT_WH, IT_WHMODE)

    def run(self, pixelA: list[torch.tensor], pixelB: list[torch.tensor],
            axis: list[str], stride: list[int], flip: list[str], width: list[int],
            height: list[int], mode: list[str], resample: list[str]) -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        pixels = pixelA + pixelB

        # make the target image
        tH = 1
        tW = 1
        target = np.zeros((tH, tW, 3), dtype=float)
        for idx, image in enumerate(pixels):

            image = comp.tensor2cv(image)
            w = width[min(idx, len(width)-1)]
            h = height[min(idx, len(height)-1)]
            rs = resample[min(idx, len(resample)-1)]
            rs = Image.Resampling[rs]
            image = comp.geo_scalefit(image, w, h, 'FIT', rs)

            # image = comp.geo_extend(image, imageB, axis[min(idx, len(axis)-1)], flip[min(idx, len(flip)-1)])
            if mode != "NONE":
                image = comp.geo_scalefit(image, w, h, mode[min(idx, len(mode)-1)], rs)

            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class ProjectionNode(JovimetrixImageInOutBaseNode):
    NAME = "🗺️ Projection (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = ""
    SORT = 55

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "proj": (["SPHERICAL", "FISHEYE"], {"default": "FISHEYE"}),
                "strength": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODEI, IT_SAMPLE)

    def run(self, pixels: list[torch.tensor], proj: list[str], strength: list[float],
            width: list[int], height: list[int], mode: list[str], invert: list[float],
            resample: list[str]) -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = comp.tensor2cv(image)
            str = strength[min(idx, len(strength)-1)]
            p = proj[min(idx, len(proj)-1)]
            logdebug(f"[REMAP] {p} ({str})")
            match p:
                case 'SPHERICAL':
                    image = comp.remap_sphere(image, str)

                case 'FISHEYE':
                    image = comp.remap_fisheye(image, str)

            rs = resample[min(idx, len(resample)-1)]
            rs = Image.Resampling[rs]
            image = comp.geo_scalefit(image,
                                  width[min(idx, len(width)-1)],
                                  height[min(idx, len(height)-1)],
                                  mode[min(idx, len(mode)-1)],
                                  rs)

            i = invert[min(idx, len(invert)-1)]
            if i != 0:
                image = comp.light_invert(image, i)

            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class CropNode(JovimetrixImageInOutBaseNode):
    NAME = "✂️ Crop (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Robust cropping with color fill"
    SORT = 55

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "top": ("FLOAT", {"default": 0, "min": 0, "step": 0.01}),
                "left": ("FLOAT", {"default": 0, "min": 0, "step": 0.01}),
                "bottom": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
                "right": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            },
            "optional": {
                "pad":  ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_PIXELS, IT_WH, d, IT_COLOR,  IT_INVERT)

    def run(self, pixels: list[torch.tensor], pad: list[bool], top: list[float],
            left: list[float], bottom: list[float], right: list[float],
            R: list[float], G: list[float], B: list[float], width: list[int],
            height: list[int], invert: list[float]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = comp.tensor2cv(image)

            t = top[min(idx, len(top)-1)]
            l = left[min(idx, len(left)-1)]
            b = bottom[min(idx, len(bottom)-1)]
            r = right[min(idx, len(right)-1)]

            color = (B[min(idx, len(B)-1)] * 255,
                     G[min(idx, len(G)-1)] * 255,
                     R[min(idx, len(R)-1)] * 255)

            w = width[min(idx, len(width)-1)]
            h = height[min(idx, len(height)-1)]
            p = pad[min(idx, len(pad)-1)]

            print(l, t, r, b, w, h, p, color)
            image = comp.geo_crop(image, l, t, r, b, w, h, p, color)

            if (i := invert[min(idx, len(invert)-1)]) != 0:
                image = comp.light_invert(image, i)

            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

# =============================================================================
# === ADJUST LUMA/COLOR NODES ===
# =============================================================================

class HSVNode(JovimetrixImageInOutBaseNode):
    NAME = "🌈 HSV (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Adjust Hue, Saturation, Value, Contrast Gamma of input."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "hue": ("FLOAT",{"default": 0, "min": 0, "max": 1, "step": 0.01},),
                "saturation": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}, ),
                "value": ("FLOAT", {"default": 1, "min": 0, "max": 250, "step": 0.01}, ),
                "contrast": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}, ),
                "gamma": ("FLOAT", {"default": 1, "min": 0, "max": 250, "step": 0.01}, ),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self, pixels: list[torch.tensor], hue: list[float], saturation: list[float],
            value: list[float], contrast: list[float], gamma: list[float],
            invert: list[float]) -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for idx, img in enumerate(pixels):
            img = comp.tensor2cv(img)

            h = hue[min(idx, len(hue)-1)]
            s = saturation[min(idx, len(saturation)-1)]
            # v = value[min(idx, len(value)-1)]
            if h != 0 or s != 0: # or v != 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                if h != 0:
                    h *= 255
                    img[:, :, 0] = (img[:, :, 0] + h) % 180

                if s != 1:
                    img[:, :, 1] = np.clip(img[:, :, 1] * s, 0, 255)

                #if v != 1:
                #    img[:, :, 2] = np.clip(img[:, :, 2] * v, 0, 255)
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            if (val := contrast[min(idx, len(contrast)-1)]) != 0:
                img = comp.light_contrast(img, 1 - val)

            #if (val := exposure[min(idx, len(exposure)-1)]) != 1:
            if (val := value[min(idx, len(value)-1)]) != 1:
                img = comp.light_exposure(img, val)

            if (val := gamma[min(idx, len(gamma)-1)]) != 1:
                img = comp.light_gamma(img, val)

            if (val := invert[min(idx, len(invert)-1)]) != 0:
                img = comp.light_invert(img, val)

            images.append(comp.cv2tensor(img))
            masks.append(comp.cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class AdjustNode(JovimetrixImageInOutBaseNode):
    NAME = "🕸️ Adjust (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Find Edges, Blur, Sharpen and Emboss an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "func": (['BLUR', 'STACK BLUR', 'GAUSSIAN BLUR', 'SHARPEN', 'EMBOSS', 'FIND EDGES', 'DILATE', 'ERODE', 'OPEN', 'CLOSE'], {"default": "BLUR"}),
            },
            "optional": {
                "radius": ("INT", {"default": 1, "min": 1,  "max": 2048, "step": 1}),
                "amount": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
                "low": ("FLOAT", {"default": 0.27, "min": 0, "max": 1, "step": 0.01}),
                "high": ("FLOAT", {"default": 0.72, "min": 0, "max": 1, "step": 0.01}),

            }}
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self, pixels: list[torch.tensor], func: list[str], radius: list[float],
            amount: list[float], low: list[float], high: list[float],
            invert: list[float])  -> tuple[torch.Tensor, torch.Tensor]:

        start = time.time()
        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = comp.tensor2cv(image)

            op = func[min(idx, len(func)-1)]
            # radius is odd for "kernels"
            rad = radius[min(idx, len(radius)-1)]
            rad = rad if rad % 2 == 1 else rad + 1
            amt = amount[min(idx, len(amount)-1)]

            lo = amount[min(idx, len(low)-1)]
            hi = amount[min(idx, len(high)-1)]

            match op:
                case 'BLUR':
                    image = cv2.blur(image, (rad, rad))

                case 'STACK BLUR':
                    image = cv2.stackBlur(image, (rad, rad))

                case 'GAUSSIAN BLUR':
                    image = cv2.GaussianBlur(image, (rad, rad), sigmaX=float(amt))

                case 'MEDIAN BLUR':
                    image = cv2.medianBlur(image, (rad, rad))

                case 'SHARPEN':
                    comp.adjust_sharpen(image, kernel_size=rad, amount=amt)

                case 'EMBOSS':
                    image = comp.morph_emboss(image, amt)

                case 'FIND_EDGES':
                    image = comp.morph_edge_detect(image, low=lo, high=hi)

                case 'OUTLINE':
                    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, (rad, rad))

                case 'DILATE':
                    image = cv2.dilate(image, (rad, rad), iterations=int(amt))

                case 'ERODE':
                    image = cv2.erode(image, (rad, rad), iterations=int(amt))

                case 'OPEN':
                    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (rad, rad))

                case 'CLOSE':
                    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (rad, rad))

            logdebug(f"[{self.NAME}] {op} {rad} {amt} {low} {hi} ({idx})")

            if (i := invert[min(idx, len(invert)-1)]) != 0:
                image = comp.light_invert(image, i)

            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        logdebug(f"[{self.NAME}] {time.time()-start:.5}")
        return (
            torch.stack(images),
            torch.stack(masks)
        )

class ThresholdNode(JovimetrixImageInOutBaseNode):
    NAME = "📉 Threshold (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Clip an input to explicit 0 or 1"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "op": ( [e.name for e in comp.EnumThreshold],
                       {"default": comp.EnumThreshold.BINARY.name}),
                "op": ( [e.name for e in comp.EnumAdaptThreshold],
                       {"default": comp.EnumAdaptThreshold.ADAPT_NONE.name}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01},),
                "block": ("INT", {"default": 3, "min": 1, "max": 101, "step": 1},),
                "const": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01},),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self, pixels: list[torch.tensor], op: list[str], adapt: list[str],
            threshold: list[float], block: list[int], const: list[float],
            invert: list[float])  -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for idx, img in enumerate(pixels):
            img = comp.tensor2cv(img)
            # force block into odd
            if block % 2 == 0:
                block += 1

            op = comp.EnumThreshold[op[min(idx, len(op)-1)]].value
            adapt = comp.EnumAdaptThreshold[adapt[min(idx, len(adapt)-1)]].value
            img = comp.adjust_threshold(img, threshold[min(idx, len(threshold)-1)], op,
                                 adapt, block[min(idx, len(block)-1)],
                                 const[min(idx, len(const)-1)])

            # img = comp.geo_scalefit(img, width[min(idx, len(width)-1)], height[min(idx, len(height)-1)], mode[min(idx, len(mode)-1)])
            if (i := invert[min(idx, len(invert)-1)]):
                img = comp.light_invert(img, i)

            images.append(comp.cv2tensor(img))
            masks.append(comp.cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class LevelsNode(JovimetrixImageInOutBaseNode):
    NAME = "📉 Level Adjust (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Clip an input based on a low, high and mid point value."
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "low": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01},),
                "mid": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01},),
                "high": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01},),
                "gamma": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01},),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODEI)

    def run(self, pixels: list[torch.tensor], low: list[float], mid: list[float],
            high: list[float], gamma: list[float], invert: list[float])  -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for idx, img in enumerate(pixels):
            img = comp.tensor2pil(img)

            l = low[min(idx, len(low)-1)]
            m = mid[min(idx, len(mid)-1)]
            h = high[min(idx, len(high)-1)]

            img = torch.maximum(img - l, torch.tensor(0.0))
            img = torch.minimum(img, (h - l))
            img = (img + m) - 0.5
            img = torch.sign(img) * torch.pow(torch.abs(img),
                                              1.0 / gamma[min(idx, len(gamma)-1)])
            img = (img + 0.5) / h

            images.append(img)
            masks.append(img)

        return (
            torch.stack(images),
            torch.stack(masks)
        )

# =============================================================================
# === COMPOSITION NODES ===
# =============================================================================

class BlendNode(JovimetrixImageInOutBaseNode):
    NAME = "⚗️ Blend (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)."
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "alpha": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            },
            "optional": {
                "func": (list(comp.OP_BLEND.keys()), {"default": "LERP"}),
                "mask": (WILDCARD, {})
        }}
        return deep_merge_dict(IT_PIXEL2, d, IT_WHMODEI)

    def run(self, pixelA: list[torch.tensor], alpha: list[float], func: list[str],
            width: list[int], height: list[int],mode: list[str], resample: list[str],
            invert: list[float], **kw) -> tuple[torch.Tensor, torch.Tensor]:

        pixelB: list[torch.tensor] = kw.get('pixelB', (torch.zeros((height[0], width[0], 3), dtype=torch.uint8),))
        mask: list[torch.tensor] = kw.get('mask', (torch.ones((height[0], width[0], 3), dtype=torch.uint8),))

        masks = []
        images = []
        for idx, image in enumerate(pixelA):
            image = comp.tensor2cv(image)
            pb = comp.tensor2cv(pixelB[min(idx, len(pixelB)-1)])
            m = comp.tensor2cv(mask[min(idx, len(mask)-1)])
            w = width[min(idx, len(width)-1)]
            h = height[min(idx, len(height)-1)]
            rs = resample[min(idx, len(resample)-1)]
            rs = Image.Resampling[rs]

            image = comp.comp_blend(image, pb, func[min(idx, len(func)-1)], w, h,
                             m, alpha[min(idx, len(alpha)-1)], rs)

            i = invert[min(idx, len(invert)-1)]
            if i != 0:
                image = comp.light_invert(image, i)
            image = comp.geo_scalefit(image, w, h, mode[min(idx, len(mode)-1)], rs)

            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

# =============================================================================
# === STREAM NODES ===
# =============================================================================

class StreamReaderNode(JovimetrixImageBaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data = list(stream.STREAMMANAGER.STREAM.keys())
        default = data[0] if len(data)-1 > 0 else ""
        d = {"required": {
                "idx": (data, {"default": default}),
                "url": ("STRING", {"default": ""}),
            },
            "optional": {
                "fps": ("INT", {"min": 1, "max": 60, "step": 1, "default": 60}),
                "hold": ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(d, IT_WHMODEI, IT_SAMPLE, IT_ORIENT, IT_CAM)

    NAME = "📺 StreamReader (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/STREAM"
    DESCRIPTION = ""

    @classmethod
    def IS_CHANGED(cls, idx: int, url: str, fps: float, hold: bool, width: int,
                   height: int, zoom: float, **kw) -> float:

        url = url if url != "" else idx
        if (device := stream.STREAMMANAGER.capture(url)) is None:
            raise Exception(f"stream failed {url}")

        if device.size[0] != width or device.size[1] != height:
            device.size = (width, height)

        if device.zoom != zoom:
            device.zoom = zoom

        if hold:
            device.pause()
            return float("nan")

        device.run()
        if device.fps != fps:
            device.fps = fps

        return float("nan")

    def run(self, idx: int, url: str, fps: float, hold: bool, width: int,
            height: int, mode: str, resample: str, invert: float, orient: str,
            zoom: float) -> tuple[torch.Tensor, torch.Tensor]:

        url = url if url != "" else idx
        if (device := stream.STREAMMANAGER.capture(url)) is None:
            raise Exception(f"stream failed {url}")

        ret, image = device.frame
        #if hold:
        #    return (comp.cv2tensor(image), comp.cv2mask(image), )

        rs = Image.Resampling[resample]
        image = comp.geo_scalefit(image, width, height, mode, rs)

        if orient in ["FLIPX", "FLIPXY"]:
            image = cv2.flip(image, 1)

        if orient in ["FLIPY", "FLIPXY"]:
            image = cv2.flip(image, 0)

        if invert != 0.:
            image = comp.light_invert(image, invert)

        return (
            torch.stack((comp.cv2tensor(image),)),
            torch.stack((comp.cv2mask(image),))
        )

class StreamWriterNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "route": ("STRING", {"default": "/stream"}),
            },
            "optional": {
                "hold": ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODEI, IT_SAMPLE, IT_ORIENT)

    NAME = "🎞️ StreamWriter (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/STREAM"
    DESCRIPTION = ""
    OUTPUT_IS_LIST = (None, None,)

    @classmethod
    def IS_CHANGED(cls, pixels: torch.Tensor, route: str, hold: bool, width: int,
                   height: int, mode: str, sampler: str, invert: float, orient: str) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super(StreamWriterNode).__init__(self, *arg, **kw)
        self.__host = None
        self.__port = None

    def run(self, pixels: torch.Tensor, route: str, hold: bool, width: int,
            height: int, mode: str, sampler: str, invert: float, orient: str) -> torch.Tensor:

       if stream.STREAMHOST != self.__host or stream.STREAMPORT != self.__port:
            # close old, if any

            # startup server
            pass

# =============================================================================
# === ANIMATE NODES ===
# =============================================================================

class TickNode(JovimetrixBaseNode):
    NAME = "🕛 Tick (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"
    DESCRIPTION = "Periodic pulse exporting normalized, delta since last pulse and count."
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", )
    RETURN_NAMES = ("🧮", "🛟", "🕛", "🔺🕛",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {},
            "optional": {
                "total": ("INT", {"min": 0, "default": 0, "step": 1}),
                # forces a MOD on total
                "loop": ("BOOLEAN", {"default": False}),
                # stick the current "count"
                "hold": ("BOOLEAN", {"default": False}),
                # manual total = 0
                "reset": ("BOOLEAN", {"default": False}),
            }}

    @classmethod
    def IS_CHANGED(cls, *arg, **kw) -> Any:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__count = 0
        # previous time, current time
        self.__time = time.time()
        self.__delta = 0

    def run(self, total: int, loop: bool, hold: bool, reset: bool) -> None:
        if reset:
            self.__count = 0

        # count = self.__count
        if loop and total > 0:
            self.__count %= total
        lin = (self.__count / total) if total != 0 else 1

        t = self.__time
        if not hold:
            self.__count += 1
            t = time.time()

        self.__delta = t - self.__time
        self.__time = t

        return (self.__count, lin, t, self.__delta,)

class WaveGeneratorNode(JovimetrixBaseNode):
    NAME = "🌊 Wave Generator (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ANIMATE"
    DESCRIPTION = ""
    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = ("🛟", "🔟", )
    POST = True

    OP_WAVE = {
        "SINE": comp.wave_sine,
        "INV SINE": comp.wave_inv_sine,
        "ABS SINE": comp.wave_abs_sine,
        "COSINE": comp.wave_cosine,
        "INV COSINE": comp.wave_inv_cosine,
        "ABS COSINE": comp.wave_abs_cosine,
        "SAWTOOTH": comp.wave_sawtooth,
        "TRIANGLE": comp.wave_triangle,
        "RAMP": comp.wave_ramp,
        "STEP": comp.wave_step_function,
        "HAVER SINE": comp.wave_haversine,
        "NOISE": comp.wave_noise,
    }
    """
        "SQUARE": comp.wave_square,
        "PULSE": comp.wave_pulse,
        "EXP": comp.wave_exponential,
        "RECT PULSE": comp.wave_rectangular_pulse,

        "LOG": comp.wave_logarithmic,
        "GAUSSIAN": comp.wave_gaussian,
        "CHIRP": comp.wave_chirp_signal,
    }
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required":{
                "wave": (list(WaveGeneratorNode.OP_WAVE.keys()), {"default": "SINE"}),
                "phase": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 1.0}),
                "amp": ("FLOAT", {"default": 0.5, "min": 0.0, "step": 0.1}),
                "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 1.0}),
                "max": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 9999.0, "step": 0.05}),
                "frame": ("INT", {"default": 1.0, "min": 0.0, "step": 1.0}),
            }}
        return d

    def run(self, wave: str, phase: float, amp: float, offset: float, max: float, frame: int) -> tuple[float, int]:
        val = 0.
        if (op := WaveGeneratorNode.OP_WAVE.get(wave, None)):
            val = op(phase, amp, offset, max, frame)
        return (val, int(val))

# =============================================================================
# === UTILITY NODES ===
# =============================================================================

class RouteNode(JovimetrixBaseNode):
    NAME = "🚌 Route (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Wheels on the BUS pass the data through, around and around."
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🚌",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
            "o": (WILDCARD, {"default": None}),
        }}

    def run(self, o: object) -> object:
        return (o,)

class ClearCacheNode(JovimetrixBaseNode):
    NAME = "🧹 Clear Cache (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Clear the torch cache, and python caches - we need to pay the bills"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🧹",)
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
            "o": (WILDCARD, {}),
        }}

    def run(self, o: Any) -> [object, ]:
        f, t = torch.cuda.mem_get_info()
        logdebug(f"[{self.NAME}] total: {t}")
        logdebug("-"* 30)
        logdebug(f"[{self.NAME}] free: {f}")

        s = o
        if isinstance(o, dict):
            s = o.copy()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        comfy.model_management.soft_empty_cache()

        f, t = torch.cuda.mem_get_info()
        logdebug(f"[{self.NAME}] free: {f}")
        logdebug("-"* 30)

        return (s, )

class OptionsNode(JovimetrixBaseNode):
    NAME = "⚙️ Options (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Change Jovimetrix Global Options"
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = ("🦄", )
    SORT = 1

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required" : {},
            "optional": {
                "o": (WILDCARD, {"default": None}),
                "log": (["ERROR", "WARN", "INFO"], {"default": "ERROR"}),
                "host": ("STRING", {"default": ""}),
                "port": ("INT", {"default": 7227}),
            }}

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def run(self, log: str, host: str, port: int, **kw) -> Any:
        if log == "ERROR":
            util.JOV_LOG = 0
        elif log == "WARN":
            util.JOV_LOG = 1
        elif log == "INFO":
            util.JOV_LOG = 2

        stream.STREAMPORT = port
        stream.STREAMHOST = host

        o = kw.get('o', None)
        return (o, )

class DisplayDataNode(JovimetrixBaseNode):
    """Display any data."""

    NAME = "📊 Display Data (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Display any data"
    SORT = 100
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
                    "source": (WILDCARD, {}),
                },
                "optional": {

                }}

    def run(self, source=None) -> dict:
        value = 'None'
        if source is not None:
            try:
                value = json.dumps(source, indent=2, sort_keys=True)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = 'source could not be serialized.'

        return {"ui": {"text": (value,)}}

class DelayNode(JovimetrixBaseNode):
    """Delay for some time."""

    NAME = "⏸️ Delay (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Delay for some time"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🦄",)
    SORT = 70

    @classmethod
    def INPUT_TYPES(cls) -> Any:
        return {"required": {
                    "o": (WILDCARD, {"default": None}),
                    "delay": ("FLOAT", {"step": 0.01, "default" : 0}),
                    "hold": ("BOOLEAN", {"default": True})
                }}

    def run(self, o: Any, delay: float, hold: bool) -> dict:
        ''' @TODO
        while hold:
            print(self)
            time.sleep(0.1)
            return ([self], )
        '''
        delay = max(0, min(delay, JOV_MAX_DELAY))
        time.sleep(delay)
        return (o,)

# =============================================================================
# === COMFYUI NODE MAP ===
# =============================================================================

import inspect
current_frame = inspect.currentframe()
calling_frame = inspect.getouterframes(current_frame)[0]
module = inspect.getmodule(calling_frame.frame)
classes = inspect.getmembers(module, inspect.isclass)

NODE_DISPLAY_NAME_MAPPINGS = {}
CLASS_MAPPINGS = {}
POST = {}
for class_name, class_object in classes:
    if class_name.endswith('Node') and not class_name.endswith('BaseNode'):
        name = class_object.NAME
        if hasattr(class_object, 'POST'):
            class_object.CATEGORY = "JOVIMETRIX 🔺🟩🔵/💣☣️ WIP ☣️💣"
            POST[name] = class_object
        else:
            CLASS_MAPPINGS[name] = class_object

# 🔗 ⚓ 🎹 📀 🍿 🎪 🐘 🤯 😱 💀 ⛓️ 🔒 🔑

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in CLASS_MAPPINGS.items()}
CLASS_MAPPINGS.update({k: v for k, v in POST.items()})

NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k, _ in POST.items()})

CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(CLASS_MAPPINGS.items(),
                                                   key=lambda item: getattr(item[1], 'SORT', 0))}
NODE_CLASS_MAPPINGS = {}

# now sort the categories...
for c in ["CREATE", "ADJUST", "TRANSFORM", "COMPOSE", "ANIMATE", "STREAM", "UTILITY", "💣☣️ WIP ☣️💣"]:
    for k, v in CLASS_MAPPINGS.items():
        if v.CATEGORY.endswith(c):
            NODE_CLASS_MAPPINGS[k] = v
            logdebug(f"{k} - {v}")
