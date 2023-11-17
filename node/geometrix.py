"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix

@author: amorano
@title: Jovimetrix Composition Pack
@nickname: Jovimetrix
@description: Shapes and Shaders.
"""

import concurrent.futures
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw

from ..util import *

# =============================================================================
# === EXPORT ===
# =============================================================================

__all__ = ["ShapeNode", "ConstantNode", "GradientNode", "GLSLShaderNode"]

# =============================================================================
# === GENERATOR ===
# =============================================================================
# Generalized supprt to make N-Gons
def sh_body(func: str, width: int, height: int, sizeX=1., sizeY=1., fill=(255, 255, 255)) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    func = getattr(d, func)
    func(xy, fill=fill)
    return image

# ellipse
def sh_ellipse(width: int, height: int, sizeX=1., sizeY=1., fill=None) -> Image:
    return sh_body('ellipse', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

# quadrilateral
def sh_quad(width: int, height: int, sizeX=1., sizeY=1., fill=None) -> Image:
    return sh_body('rectangle', width, height, sizeX=sizeX, sizeY=sizeY, fill=fill)

# polygon
def sh_polygon(width: int, height: int, size: float=1., sides: int=3, angle: float=0., fill=None) -> Image:
    fill=fill or (255, 255, 255)
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), 'black')
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image

#
class ShapeNode:
    @classmethod
    def INPUT_TYPES(s):
        d = {
            "required": {
                "shape": (["CIRCLE", "SQUARE", "ELLIPSE", "RECTANGLE", "POLYGON"], {"default": "SQUARE"}),
                "sides": ("INT", {"default": 3, "min": 3, "max": 100, "step": 1}),
            },
        }
        return deep_merge_dict(d, IT_WH, IT_COLOR, IT_ROT, IT_SCALE, IT_INVERT)

    DESCRIPTION = ""
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, shape, sides, width, height, R, G, B, angle, sizeX, sizeY, invert):
        image = None
        fill = (int(R * 255.),
                int(G * 255.),
                int(B * 255.),)

        match shape:
            case 'SQUARE':
                image = sh_quad(width, height, sizeX, sizeX, fill=fill)

            case 'ELLIPSE':
                image = sh_ellipse(width, height, sizeX, sizeY, fill=fill)

            case 'RECTANGLE':
                image = sh_quad(width, height, sizeX, sizeY, fill=fill)

            case 'POLYGON':
                image = sh_polygon(width, height, sizeX, sides, fill=fill)

            case _:
                image = sh_ellipse(width, height, sizeX, sizeX, fill=fill)

        image = image.rotate(-angle)
        image = pil2cv(image)
        if invert > 0.:
            image = INVERT(image, invert)
        return (cv2tensor(image),)

# =============================================================================
# === CONSTANT NODE ===
# =============================================================================
class ConstantNode:
    @classmethod
    def INPUT_TYPES(s):
        return deep_merge_dict(IT_WH, IT_COLOR)

    DESCRIPTION = ""
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, width, height, R, G, B):
        image = Image.new("RGB", (width, height), (int(R * 255.), int(G * 255.), int(B * 255.)) )
        return (pil2tensor(image),)

# =============================================================================
# === GRADIENT NODE ===
# =============================================================================
def gradient_radial(width: int, height: int, center: tuple = None, radius: float = None,
                    color1: tuple = (255, 255, 255), color2: tuple = (0, 0, 0)) -> np.ndarray:

    if center:
        center = np.clip(center, -0.5, 0.5)
        center = (width * (0.5 + center[0]), height * (0.5 + center[1]))
    else:
        center = (width // 2, height // 2)

    if radius:
        radius = np.clip(radius, 0, 1)
        radius = min(width, height) / 2. * radius
    else:
        radius = min(width, height) / 2.

    # Create a meshgrid for calculating the distance from the center
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Convert color1 and color2 to arrays and reshape them for proper broadcasting
    color1 = np.array(color1).reshape(1, 1, 3)
    color2 = np.array(color2).reshape(1, 1, 3)

    # Create the gradient image
    gradient_image = np.zeros((height, width, 3), dtype=np.uint8)
    a = np.maximum(0, color1 * (1 - distance / radius)[:, :, np.newaxis])
    b = np.maximum(0, color2 * (distance / radius)[:, :, np.newaxis])
    gradient_image[:, :] = (a + b).astype(np.uint8)
    return gradient_image

class Point(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def rot_x(self, degrees):
        radians = np.radians(degrees)
        return self.x * np.cos(radians) + self.y * np.sin(radians)

def gradient_color(minval, maxval, val, color_palette: List[Tuple]) -> tuple[int, int, int]:
    """ Computes intermediate RGB color of a value in the range of minval
        to maxval (inclusive) based on a color_palette representing the range.
    """
    max_index = len(color_palette)-1
    delta = maxval - minval
    if delta == 0:
        delta = 1
    v = float(val-minval) / delta * max_index
    i1, i2 = int(v), min(int(v)+1, max_index)
    (r1, g1, b1), (r2, g2, b2) = color_palette[i1], color_palette[i2]
    f = v - i1
    return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

def gradient_linear(width: int, height: int, color_palette: List[Tuple], degrees: float=0.) -> np.ndarray:
    minval, maxval = 1, len(color_palette)
    delta = maxval - minval
    # make a 32x32 and stretch
    size = 32
    image = Image.new("RGB", (size, size), 'WHITE')
    for x in range(size):
        for y in range(size):
            p = Point(x, y)
            f = (p.rot_x(degrees)) / size
            val = minval + f * delta
            color = gradient_color(minval, maxval, val, color_palette)
            image.putpixel((x, y), color)

    image = image.resize((width, height))
    return pil2np(image)

class GradientNode:
    @classmethod
    def INPUT_TYPES(s):
        d = {
            "required": {
                "style": (["LINEAR", "RADIAL", "DIAMOND"], {"default": "LINEAR"}),
            },
            "optional": {
                "angle": ("FLOAT", {"default": 0., "min": -180., "max": 180., "step": 10}),
                "centerX": ("FLOAT", {"default": 0., "min": -0.5, "max": 0.5, "step": 0.1}),
                "centerY": ("FLOAT", {"default": 0., "min": -0.5, "max": 0.5, "step": 0.1}),
                "radius": ("FLOAT", {"default": 1., "min": 0., "max": 2., "step": 0.1}),
            }
        }
        return deep_merge_dict(d, IT_WH)

    DESCRIPTION = ""
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, style, angle, width, height, centerX, centerY, radius):
        angle = np.deg2rad(angle)
        match style:
            case "RADIAL":
                image = gradient_radial(width, height, (centerX, centerY), radius=radius, )

            case "DIAMOND":
                image = np.ones((height, width, 3), dtype=np.uint8)

            case _:
                palette = [
                    (0,0,0),
                    (255,255,255)
                ]
                image = gradient_linear(width, height, palette, angle)

        return (np2tensor(image),)

# =============================================================================
# === PER PIXEL SHADER NODE ===
# =============================================================================
class PixelShaderNode:
    @classmethod
    def INPUT_TYPES(s):
        d = {
            "required": {},
            "optional": {
                "R": ("STRING", {"multiline": True, "default": "1. - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 2))"}),
                "G": ("STRING", {"multiline": True}),
                "B": ("STRING", {"multiline": True}),
            },
        }
        return deep_merge_dict(d, IT_WH)

    DESCRIPTION = ""
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE",)
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, width, height, R, G, B):
        import math
        from ast import literal_eval

        # Create an empty numpy array to store the pixel values
        image = np.zeros((height, width, 3), dtype=np.uint8)

        R = R.lower().strip()
        G = G.lower().strip()
        B = B.lower().strip()

        def parseChannel(chan, x, y, tu, tv, w, h) -> str:
            exp = chan.replace("$x", str(x))
            exp = exp.replace("$y", str(y))
            exp = exp.replace("$u", str(tu))
            exp = exp.replace("$v", str(tv))
            exp = exp.replace("$w", str(w))
            exp = exp.replace("$h", str(h))
            return exp

        # Define the pixel shader function
        def pixel_shader(x, y, tu, tv, w, h):
            result = []
            for who in (B, G, R, ):
                if who == "":
                    result.append(0)
                    continue

                exp = parseChannel(who, x, y, tu, tv, w, h)
                try:
                    val = literal_eval(exp)
                except:
                    try:
                        val = eval(exp.replace("^", "**"))
                    except Exception as e:
                        #print(str(e))
                        continue
                result.append(int(val * 255))
            return result

        # Function to process a chunk in parallel
        def process_chunk(chunk_coords):
            y_start, y_end, x_start, x_end, width, height = chunk_coords
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    image[y, x] = pixel_shader(x, y, x/width, y/height, width, height)

        # 12 seems to be the legit balance
        chunkX = chunkY = 12

        # Divide the image into chunks
        chunk_coords = []
        for y in range(0, height, chunkY):
            for x in range(0, width, chunkX):
                y_end = min(y + chunkY, height)
                x_end = min(x + chunkX, width)
                chunk_coords.append((y, y_end, x, x_end, width, height))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunk_coords}
            for future in concurrent.futures.as_completed(futures):
                pass

        return (pil2tensor(cv2pil(image)),)

NODE_CLASS_MAPPINGS = {
    "✨ Shape Generator (jov)": ShapeNode,
    "🔆 Pixel Shader Image (jov)": PixelShaderNode,
    "🟪 Constant Image (jov)": ConstantNode,
    #"🥻 Gradient (jov)": GradientNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
