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

import gc
import re
import json
import time
from typing import Any

import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter

try:
    from .sup.util import LOGLEVEL, loginfo, logwarn, logerr
    from .sup.stream import STREAMHOST, STREAMPORT, STREAMMANAGER
    from .sup import comp
    from .sup.anim import ease, EaseOP
except:

    from sup.util import LOGLEVEL, loginfo, logwarn, logerr
    from sup.stream import STREAMHOST, STREAMPORT, STREAMMANAGER
    import sup.comp as comp
    from sup.anim import ease, EaseOP

# =============================================================================
# === CORE NODES ===
# =============================================================================

class JovimetrixBaseNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required":{}}

    NAME = "Jovimetrix"
    DESCRIPTION = "A Jovimetrix Node"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ()
    OUTPUT_NODE = False
    INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (True, )
    FUNCTION = "run"

class JovimetrixImageBaseNode(JovimetrixBaseNode):
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
    "required": {
        "pixels": (WILDCARD, {"default": None}),
    }}

IT_PIXEL2 = {
    "required": {
        "pixelA": (WILDCARD, {"default": None}),
        "pixelB": (WILDCARD, {"default": None}),
    }}

IT_WH = {
    "optional": {
        "width": ("INT", {"default": 256, "min": 32, "max": 8192, "step": 1, "display": "number"}),
        "height": ("INT", {"default": 256, "min": 32, "max": 8192, "step": 1, "display": "number"}),
    }}

IT_WHMODE = {
    "optional": {
        "mode": (["NONE", "FIT", "CROP", "ASPECT"], {"default": "NONE"}),
        "resample": ([e.name for e in Image.Resampling], {"default": Image.Resampling.LANCZOS.name}),
    }}

IT_TRANS = {
    "optional": {
        "offsetX": ("FLOAT", {"default": 0., "min": -1., "max": 1., "step": 0.01, "display": "number"}),
        "offsetY": ("FLOAT", {"default": 0., "min": -1., "max": 1., "step": 0.01, "display": "number"}),
    }}

IT_ROT = {
    "optional": {
        "angle": ("FLOAT", {"default": 0., "min": -180., "max": 180., "step": 1., "display": "number"}),
    }}

IT_SCALE = {
    "optional": {
        "sizeX": ("FLOAT", {"default": 1., "min": 0.01, "max": 2., "step": 0.01, "display": "number"}),
        "sizeY": ("FLOAT", {"default": 1., "min": 0.01, "max": 2., "step": 0.01, "display": "number"}),
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
        "invert": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": 0.01}),
    }}

IT_COLOR = {
    "optional": {
        "R": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01, "display": "number"}),
        "G": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01, "display": "number"}),
        "B": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01, "display": "number"}),
    }}

IT_ORIENT = {
    "optional": {
        "orient": (["NORMAL", "FLIPX", "FLIPY", "FLIPXY"], {"default": "NORMAL"}),
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
            image = comp.INVERT(image, invert)
            image = comp.cv2pil(image)

        return (comp.pil2tensor(image), comp.pil2tensor(image.convert("L")), )

class PixelShaderBaseNode(JovimetrixImageBaseNode):
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {},
            "optional": {
                "R": ("STRING", {"multiline": True, "default": "255 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5)) * 255"}),
                "G": ("STRING", {"multiline": True, "default": "255 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5)) * 255"}),
                "B": ("STRING", {"multiline": True, "default": "255 - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 3.5)) * 255"}),
            },
        }
        if cls == PixelShaderImageNode:
            return deep_merge_dict(IT_IMAGE, d, IT_WH, IT_SAMPLE)
        return deep_merge_dict(d, IT_WH)

    @staticmethod
    def shader(image: cv2.Mat, R: str, G: str, B: str, width: int, height: int, chunkX: int=64, chunkY:int=64) -> np.ndarray:
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
                    "$r": image[y, x, 2],
                    "$g": image[y, x, 1],
                    "$b": image[y, x, 0],
                }

                parseR = re.sub(r'\$(\w+)', lambda match: str(variables.get(match.group(0), match.group(0))), R)
                parseG = re.sub(r'\$(\w+)', lambda match: str(variables.get(match.group(0), match.group(0))), G)
                parseB = re.sub(r'\$(\w+)', lambda match: str(variables.get(match.group(0), match.group(0))), B)

                result = []
                for i, rgb in enumerate([parseB, parseG, parseR]):
                    if rgb == "":
                        out[y, x, i] = image[y, x, i]
                        continue

                    try:
                        out[y, x, i]  = literal_eval(rgb)
                    except:
                        try:
                            out[y, x, i] = eval(rgb.replace("^", "**"))
                        except Exception as e:
                            if not err:
                                err = True
                                logerr(f'eval failed {str(e)}\n{parseR}\n{parseG}\n{parseB}')

        return np.clip(out, 0, 255).astype(np.uint8)

    def run(self, image: torch.tensor, R: str, G: str, B: str, width: int, height: int, resample: str) -> tuple[torch.Tensor, torch.Tensor]:
        resample = Image.Resampling[resample]
        image = comp.tensor2cv(image)
        image = cv2.resize(image, (width, height), interpolation=resample)
        image = PixelShaderBaseNode.shader(image, R, G, B, width, height)
        return (comp.cv2tensor(image), comp.cv2mask(image), )

class PixelShaderNode(PixelShaderBaseNode):
    NAME = "🔆 Pixel Shader (jov)"
    DESCRIPTION = ""

    def run(self,  R: str, G: str, B: str, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.zeros((height, width, 3), dtype=torch.uint8)
        return super().run(image, R, G, B, width, height, Image.Resampling.LANCZOS.name)

class PixelShaderImageNode(PixelShaderBaseNode):
    NAME = "🔆 Pixel Shader Image (jov)"
    DESCRIPTION = ""

class GLSLNode(JovimetrixImageBaseNode):
    NAME = "🍩 GLSL (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/CREATE"
    DESCRIPTION = ""
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
# === TRANFORM NODES ===
# =============================================================================

class TransformNode(JovimetrixImageBaseNode):
    NAME = "🌱 Transform (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Translate, Rotate, Scale, Tile and Invert an input. CROP or WRAP the edges."
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_PIXELS, IT_TRS, IT_EDGE, IT_WH, IT_WHMODE)

    def run(self, pixels: list[torch.tensor], offsetX: list[float],
            offsetY: list[float], angle: list[float], sizeX: list[float],
            sizeY: list[float], edge: list[str], width: list[int], height: list[int],
            mode: list[str], resample: list[str]) -> tuple[torch.Tensor, torch.Tensor]:

        loginfo(resample)
        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = comp.tensor2cv(image)
            image = comp.TRANSFORM(image,
                                   offsetX[min(idx, len(offsetX))],
                                   offsetY[min(idx, len(offsetY))],
                                   angle[min(idx, len(angle))],
                                   sizeX[min(idx, len(sizeX))],
                                   sizeY[min(idx, len(sizeY))],
                                   edge[min(idx, len(edge))],
                                   width[min(idx, len(width))],
                                   height[min(idx, len(height))],
                                   mode[min(idx, len(mode))],
                                   resample[min(idx, len(resample))],
                                )
            images.append(comp.cv2tensor(image))
            masks.append(comp.cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class TileNode(JovimetrixImageBaseNode):
    NAME = "🔳 Tile (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Tile an Image with optional crop to original image size."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_PIXELS, IT_TILE)

    def run(self, pixels: torch.tensor, tileX: float, tileY: float) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = comp.tensor2cv(pixels)
        height, width, _ = pixels.shape
        pixels = comp.EDGEWRAP(pixels, tileX, tileY)
        # rebound to target width and height
        pixels = cv2.resize(pixels, (width, height), interpolation=Image.LANCZOS)
        return (comp.cv2tensor(pixels), comp.cv2mask(pixels), )

class MirrorNode(JovimetrixImageBaseNode):
    NAME = "🔰 Mirror (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Flip an input across the X axis, the Y Axis or both, with independent centers."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "x": ("FLOAT", {"default": 0.5, "min": 0., "max": 1., "step": 0.01}),
                "y": ("FLOAT", {"default": 0.5, "min": 0., "max": 1., "step": 0.01}),
                "mode": (["X", "Y", "XY", "YX"], {"default": "X"}),
            },
        }
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self, pixels, x, y, mode, invert) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = comp.tensor2cv(pixels)
        while (len(mode) > 0):
            axis, mode = mode[0], mode[1:]
            px = [y, x][axis == 'X']
            pixels = comp.MIRROR(pixels, px, int(axis == 'X'), invert=invert)
        return (comp.cv2tensor(pixels), comp.cv2mask(pixels), )

class ExtendNode(JovimetrixImageBaseNode):
    NAME = "🎇 Extend (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Contrast, Gamma and Exposure controls for images."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "axis": (["HORIZONTAL", "VERTICAL"], {"default": "HORIZONTAL"}),
            },
            "optional": {
                "flip": ("BOOLEAN", {"default": False}),
            },
        }
        return deep_merge_dict(IT_PIXEL2, d, IT_WH, IT_WHMODE)

    def run(self, pixelA: torch.tensor, pixelB: torch.tensor, axis: str, flip: str,
            width: int, height: int, mode: str, resample: str) -> tuple[torch.Tensor, torch.Tensor]:

        resample = Image.Resampling[resample]
        pixelA = comp.SCALEFIT(comp.tensor2cv(pixelA), width, height, 'FIT', resample)
        pixelB = comp.SCALEFIT(comp.tensor2cv(pixelB), width, height, 'FIT', resample)

        pixelA = comp.EXTEND(pixelA, pixelB, axis, flip)
        if mode != "NONE":
            pixelA = comp.SCALEFIT(pixelA, width, height, mode, resample)
        return (comp.cv2tensor(pixelA), comp.cv2mask(pixelA), )

class ProjectionNode(JovimetrixImageBaseNode):
    NAME = "🗺️ Projection (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = ""
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "proj": (["SPHERICAL", "CYLINDRICAL"], {"default": "SPHERICAL"}),
            }}
        return deep_merge_dict(IT_IMAGE, d, IT_WH)

    def run(self, image: torch.tensor, proj: str, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = comp.tensor2pil(image)

        source_width, source_height = image.size
        target_image = Image.new("RGB", (width, height))
        for y_target in range(height):
            for x_target in range(width):
                x_source = int((x_target / width) * source_width)

                if proj == "SPHERICAL":
                    x_source %= source_width
                y_source = int(y_target / height * source_height)
                px = image.getpixel((x_source, y_source))

                target_image.putpixel((x_target, y_target), px)
        return (comp.pil2tensor(target_image), comp.pil2mask(target_image),)

# =============================================================================
# === ADJUST LUMA/COLOR NODES ===
# =============================================================================

class HSVNode(JovimetrixImageBaseNode):
    NAME = "🌈 HSV (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Adjust Hue, Saturation, Value, Gamma, Contrast and Exposure of an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "hue": ("FLOAT",{"default": 0., "min": 0., "max": 1., "step": 0.01},),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0., "max": 2., "step": 0.01}, ),
                "value": ("FLOAT", {"default": 1.0, "min": 0., "max": 100., "step": 0.01}, ),
                "contrast": ("FLOAT", {"default": 0., "min": 0., "max": 2., "step": 0.01}, ),
                "exposure": ("FLOAT", {"default": 1.0, "min": 0., "max": 2., "step": 0.01}, ),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0., "max": 2., "step": 0.01}, ),
            }}
        return deep_merge_dict(IT_IMAGE, d, IT_INVERT)

    def run(self, image: torch.tensor, hue: float, saturation: float, value: float, contrast: float,
            exposure: float, gamma: float, invert: float) -> tuple[torch.Tensor, torch.Tensor]:

        # loginfo(1, image.dtype, hue, saturation, value, contrast, exposure, gamma, invert)
        image = comp.tensor2cv(image)
        if hue != 0 or saturation != 1 or value != 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if hue != 0:
                hue *= 255
                image[:, :, 0] = (image[:, :, 0] + hue) % 180

            if saturation != 1:
                image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)

            if value != 1:
                image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        if contrast != 0:
            image = comp.CONTRAST(image, contrast)

        if exposure != 1:
            image = comp.EXPOSURE(image, exposure)

        if gamma != 1:
            image = comp.GAMMA(image, gamma)

        if invert != 0:
            image = comp.INVERT(image, invert)

        return (comp.cv2tensor(image), comp.cv2mask(image), )

class AdjustNode(JovimetrixImageBaseNode):
    NAME = "🕸️ Adjust (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Find Edges, Blur, Sharpen and Emboss an input"

    OPS = {
        'BLUR': ImageFilter.GaussianBlur,
        'SHARPEN': ImageFilter.UnsharpMask,
    }

    OPS_PRE = {
        # PREDEFINED
        'EMBOSS': ImageFilter.EMBOSS,
        'FIND_EDGES': ImageFilter.FIND_EDGES,
    }

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        ops = list(AdjustNode.OPS.keys()) + list(AdjustNode.OPS_PRE.keys())
        d = {"required": {
                "func": (ops, {"default": "BLUR"}),
                "radius": ("INT", {"default": 1, "min": 0, "step": 1}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self, pixels: torch.tensor, func: str, radius: float, invert: float)  -> tuple[torch.Tensor, torch.Tensor]:
        pixels = comp.tensor2pil(pixels)

        if (op := AdjustNode.OPS.get(func, None)):
            pixels = pixels.filter(op(radius))
        elif (op := AdjustNode.OPS_PRE.get(func, None)):
            pixels = pixels.filter(op())

        if invert != 0:
            pixels = comp.pil2cv(pixels)
            pixels = comp.INVERT(pixels, invert)
            pixels = comp.cv2pil(pixels)
        return (comp.pil2tensor(pixels), comp.pil2tensor(pixels.convert("L")), )

class ThresholdNode(JovimetrixImageBaseNode):
    NAME = "📉 Threshold (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Clip an input based on a mid point value"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "op": ( [e.name for e in comp.EnumThreshold], {"default": comp.EnumThreshold.BINARY.name}),
                "op": ( [e.name for e in comp.EnumAdaptThreshold], {"default": comp.EnumAdaptThreshold.ADAPT_NONE.name}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0., "max": 1., "step": 0.01},),
                "block": ("INT", {"default": 3, "min": 1, "max": 101, "step": 1},),
                "const": ("FLOAT", {"default": 0, "min": -1., "max": 1., "step": 0.01},),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODEI)

    def run(self, pixels: torch.tensor, op: str, adapt: str, threshold: float,
            block: int, const: float, width: int, height: int, mode: str, invert: float)  -> tuple[torch.Tensor, torch.Tensor]:

        pixels = comp.tensor2cv(pixels)
        # force block into odd
        if block % 2 == 0:
            block += 1

        op = comp.EnumThreshold[op].value
        adapt = comp.EnumAdaptThreshold[adapt].value
        pixels = comp.THRESHOLD(pixels, threshold, op, adapt, block, const)
        pixels = comp.SCALEFIT(pixels, width, height, mode)
        if invert:
            pixels = comp.INVERT(pixels)
        return (comp.cv2tensor(pixels), comp.cv2mask(pixels), )

# =============================================================================
# === COMPOSITION NODES ===
# =============================================================================

class BlendBaseNode(JovimetrixImageBaseNode):
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "alpha": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
            },
            "optional": {
                "func": (list(comp.OP_BLEND.keys()), {"default": "LERP"}),
        }}

        if cls == BlendMaskNode:
            e = {"optional": {
                    "mask": (WILDCARD, {"default": None})
                }}
            return deep_merge_dict(IT_PIXEL2, e, d, IT_WHMODEI)
        return deep_merge_dict(IT_PIXEL2, d, IT_WHMODEI)

    def run(self, pixelA: torch.tensor, pixelB: torch.tensor, alpha: float, func: str, mask: torch.tensor,
            width: int, height: int, mode: str, resample: str, invert: float) -> tuple[torch.Tensor, torch.Tensor]:

        resample = Image.Resampling[resample]
        pixelA = comp.tensor2cv(pixelA)
        pixelB = comp.tensor2cv(pixelB)
        mask = comp.tensor2cv(mask)
        pixelA = comp.BLEND(pixelA, pixelB, func, width, height, mask=mask, alpha=alpha)
        if invert:
            pixelA = comp.INVERT(pixelA, invert)
        pixelA = comp.SCALEFIT(pixelA, width, height, mode, resample)
        return (comp.cv2tensor(pixelA), comp.cv2mask(pixelA),)

class BlendNode(BlendBaseNode):

    NAME = "⚗️ Blend (jov)"
    DESCRIPTION = "Applies selected operation to 2 inputs with using a linear blend (alpha)."

    def run(self, pixelA: torch.tensor, pixelB: torch.tensor, alpha: float, func: str,
            width: int, height: int, mode: str, resample: str, invert: float) -> tuple[torch.Tensor, torch.Tensor]:

        mask = torch.ones((height, width))
        return super().run(pixelA, pixelB, alpha, func, mask, width, height, mode, resample, invert)

class BlendMaskNode(BlendBaseNode):

    NAME = "⚗️ Blend Mask (jov)"
    DESCRIPTION = "Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)."

    def run(self, pixelA: torch.tensor, pixelB: torch.tensor, alpha: float, func: str, mask: torch.tensor,
            width: int, height: int, mode: str, resample: str, invert: float) -> tuple[torch.Tensor, torch.Tensor]:

        return super().run(pixelA, pixelB, alpha, func, mask, width, height, mode, resample, invert)

# =============================================================================
# === STREAM NODES ===
# =============================================================================

class StreamReaderNode(JovimetrixImageBaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        data = list(STREAMMANAGER.STREAM.keys())
        default = data[0] if len(data) > 0 else ""
        d = {"required": {
                "idx": (data, {"default": default}),
                "url": ("STRING", {"default": ""}),
                "fps": ("INT", {"min": 1, "max": 60, "step": 1, "default": 60}),
            },
            "optional": {
                "hold": ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(d, IT_WHMODEI, IT_ORIENT)

    NAME = "📺 StreamReader (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/STREAM"
    DESCRIPTION = ""

    @classmethod
    def IS_CHANGED(cls, idx: int, url: str, fps: float, hold: bool, width: int,
                   height: int, mode: str, resample: str, invert: float, orient: str) -> float:
        url = url if url != "" else idx
        if (stream := STREAMMANAGER.capture(url)) is None:
            raise Exception(f"stream failed {url}")

        if stream.size[0] != width or stream.size[1] != height:
            stream.size = (width, height)

        if stream.fps != fps:
            stream.fps = fps

        if hold:
            stream.pause()
        else:
            stream.run()

        return float("nan")

    def run(self, idx: int, url: str, fps: float, hold: bool, width: int,
            height: int, mode: str, resample: str, invert: float, orient: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a current frame from the StreamReader if it is active and the FPS check passes.

        Args:
            idx (int): Index of the StreamReader.
            url (str): URI for a streaming device.
            fps (float): Frames per second.
            hold (bool): Hold last frame flag.
            width (int): Width of the image.
            height (int): Height of the image.
            mode (str): Scale processing mode.
            invert (float): Amount to invert the output
            orient (str): Normal, FlipX, FlipY or FlipXY

        Returns:
            (image (torch.tensor), mask (torch.tensor)): The image and its mask result.
        """

        _, image = STREAMMANAGER.frame(idx)
        if hold:
            return (comp.cv2tensor(image), comp.cv2mask(image), )

        image = comp.SCALEFIT(image, width, height, mode, resample)

        if orient in ["FLIPX", "FLIPXY"]:
            image = cv2.flip(image, 1)

        if orient in ["FLIPY", "FLIPXY"]:
            image = cv2.flip(image, 0)

        if invert != 0.:
            image = comp.INVERT(image, invert)

        return (comp.cv2tensor(image), comp.cv2mask(image), )

class StreamWriterNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "route": ("STRING", {"default": "/stream"}),
            },
            "optional": {
                "hold": ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODEI, IT_ORIENT)

    NAME = "🎞️ StreamWriter (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/STREAM"
    DESCRIPTION = ""

    @classmethod
    def IS_CHANGED(cls, pixels: torch.Tensor, route: str, hold: bool, width: int, height: int, mode: str, invert: float, orient: str) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super(StreamWriterNode).__init__(self, *arg, **kw)
        self.__host = None
        self.__port = None

    def run(self, pixels: torch.Tensor, route: str, hold: bool, width: int, height: int, mode: str, invert: float, orient: str) -> torch.Tensor:
        if STREAMHOST != self.__host or STREAMPORT != self.__port:
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
    RETURN_NAMES = ("count 🧮", "0-1", "time", "frame 🛆",)

    OUTPUT_NODE = True
    # INPUT_IS_LIST = True
    # OUTPUT_IS_LIST = (False, )

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
    INPUT_IS_LIST = (True, )
    OUTPUT_IS_LIST = (True, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
            "o": (WILDCARD, {"default": None}),
        }}

    def run(self, o: list[object]) -> list[object]:
        return (o,)

class ClearCacheNode(JovimetrixBaseNode):
    NAME = "🧹 Clear Global Cache (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Clear the torch cache, and python caches - we need to pay the bills"
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("🧹",)
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {
            "o": (AnyType("*"), {"default": None}),
        }}

    @classmethod
    def IS_CHANGED(cls) -> float:
        return float("nan")

    def run(self, o: torch.Tensor) -> [object, ]:
        s = o.copy()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return (s,)

class OptionsNode(JovimetrixBaseNode):
    NAME = "⚙️ Options (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Change Jovimetrix Global Options"
    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = ("", )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "o": (WILDCARD, {"default": None}),
                "log": (["ERROR", "WARN", "INFO"], {"default": "ERROR"}),
                "host": (["STRING"], {"default": ""}),
                "port": (["INT"], {"default": 7227}),
            }}

    def run(self, o: Any, log: str, host: str, port: int) -> tuple[torch.Tensor, torch.Tensor]:
        global LOGLEVEL
        if log == "ERROR":
            LOGLEVEL = 0
        elif log == "WARN":
            LOGLEVEL = 1
        elif log == "INFO":
            LOGLEVEL = 2
        print(LOGLEVEL)
        return (o, )

class DisplayDataNode(JovimetrixBaseNode):
    """Display any data node."""

    NAME = "Display Data"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = "Change Jovimetrix Global Options"
    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
            "source": (WILDCARD, {}),
            },
        }

    def main(self, source=None) -> dict:
        value = 'None'
        if source is not None:
            try:
                value = json.dumps(source)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = 'source exists, but could not be serialized.'

        return {"ui": {"text": (value,)}}

# =============================================================================
# === COMFYUI NODE MAP ===
# =============================================================================

import inspect
current_frame = inspect.currentframe()
calling_frame = inspect.getouterframes(current_frame)[0]
module = inspect.getmodule(calling_frame.frame)
classes = inspect.getmembers(module, inspect.isclass)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS = {}
POST = {}
for class_name, class_object in classes:
    if class_name.endswith('Node') and not class_name.endswith('BaseNode'):
        name = class_object.NAME #.encode('utf-8')
        if hasattr(class_object, 'POST'):
            class_object.CATEGORY = "JOVIMETRIX 🔺🟩🔵/**WIP**"
            POST[name] = class_object
        else:
            NODE_CLASS_MAPPINGS[name] = class_object
        cat = class_object.CATEGORY.split('/')[-1].strip(']')
        loginfo(f"({cat}) {name}")

# 🔗 ⚓ 🎹 📀 🍿 🎪 🐘

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in NODE_CLASS_MAPPINGS.items()}
NODE_CLASS_MAPPINGS.update({k: v for k, v in POST.items()})
NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k, _ in POST.items()})
