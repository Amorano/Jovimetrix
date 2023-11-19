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

import cv2
import torch
import concurrent.futures
import numpy as np

from .. import IT_WH, deep_merge_dict
from ..util import JovimetrixBaseNode, cv2mask, cv2tensor, tensor2cv

__all__ = ["PixelShaderNode", "PixelShaderImageNode"]

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

class PixelShaderBaseNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        d = {"optional": {
                "R": ("STRING", {"multiline": True, "default": "1. - np.minimum(1, np.sqrt((($u-0.5)**2 + ($v-0.5)**2) * 2))"}),
                "G": ("STRING", {"multiline": True}),
                "B": ("STRING", {"multiline": True}),
            },
        }
        return deep_merge_dict(d, IT_WH)

    def run(self, image, width, height, R, G, B):
        image = tensor2cv(image)
        image = shader(image, width, height, R, G, B)
        return (cv2tensor(image), cv2mask(image), )

class PixelShaderNode(PixelShaderBaseNode):
    DESCRIPTION = ""
    def run(self, width, height, R, G, B):
        image = torch.empty((height, width, 3), dtype=np.uint8)
        return super().run(image, width, height, R, G, B)

class PixelShaderImageNode(PixelShaderBaseNode):
    DESCRIPTION = ""

NODE_CLASS_MAPPINGS = {

    "🔆 Pixel Shader (jov)": PixelShaderNode,
    "🔆 Pixel Shader Image (jov)": PixelShaderImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
