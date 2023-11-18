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
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("SHAPE", "MASK", )
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
        if invert > 0.:
            image = pil2cv(image)
            image = INVERT(image, invert)
            image = cv2pil(image)

        return (pil2tensor(image), pil2tensor(image.convert("L")), )

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
# === PER PIXEL SHADER NODE ===
# =============================================================================
def shader(image: cv2.Mat, width: int, height: int, R: str, G: str, B: str):
    import math
    from ast import literal_eval

    R = R.lower().strip()
    G = G.lower().strip()
    B = B.lower().strip()

    def parseChannel(chan, x, y, u, v, i, w, h) -> str:
        """
        x, y - current x,y position (output)
        u, v - tex-coord position (output)
        w, h - width/height (output)
        i    - value in original image at (x, y)
        """
        exp = chan.replace("$x", str(x))
        exp = exp.replace("$y", str(y))
        exp = exp.replace("$u", str(u))
        exp = exp.replace("$v", str(v))
        exp = exp.replace("$w", str(w))
        exp = exp.replace("$h", str(h))
        ir, ig, ib, = i
        exp = exp.replace("$r", str(ir))
        exp = exp.replace("$g", str(ig))
        exp = exp.replace("$b", str(ib))
        return exp

    # Define the pixel shader function
    def pixel_shader(x, y, u, v, w, h):
        result = []
        i = image[y, x]
        for who, val in ((B, i[2]), (G, i[1]), (R, i[0]), ):
            if who == "":
                result.append(val)
                continue
            exp = parseChannel(who, x, y, u, v, val, w, h)
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

    # 12 seems to be the legit balance *for single node
    chunkX = chunkY = 8

    # Divide the image into chunks
    chunk_coords = []
    for y in range(0, height, chunkY):
        for x in range(0, width, chunkX):
            y_end = min(y + chunkY, height)
            x_end = min(x + chunkX, width)
            chunk_coords.append((y, y_end, x, x_end, width, height))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunk_coords}
        for _ in concurrent.futures.as_completed(futures):
            pass

    return image

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
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("SHAPE", "MASK", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, width, height, R, G, B):
        # Create an empty numpy array to store the pixel values
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image = shader(image, width, height, R, G, B)
        # print('PixelShaderImageNode', image.shape)
        return (cv2tensor(image), cv2mask(image), )

class PixelShaderImageNode:
    @classmethod
    def INPUT_TYPES(s):
        d = {
            "required": {},
            "optional": {
                "image": ("IMAGE", ),
                "R": ("STRING", {"multiline": True, "default": "1. - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 2))"}),
                "G": ("STRING", {"multiline": True}),
                "B": ("STRING", {"multiline": True}),
            },
        }
        return deep_merge_dict(d, IT_WH)

    DESCRIPTION = ""
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("SHAPE", "MASK", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, width, height, R, G, B):
        image = tensor2cv(image)
        h, w, _ = image.shape
        if h != height or w != width:
            # force input image to desired output for sampling
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = shader(image, width, height, R, G, B)
        # print('PixelShaderImageNode', image.shape)
        return (cv2tensor(image), cv2mask(image), )

NODE_CLASS_MAPPINGS = {
    "✨ Shape Generator (jov)": ShapeNode,
    "🔆 Pixel Shader (jov)": PixelShaderNode,
    "🔆 Pixel Shader Image (jov)": PixelShaderImageNode,
    "🟪 Constant Image (jov)": ConstantNode,
    #"🥻 Gradient (jov)": GradientNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
