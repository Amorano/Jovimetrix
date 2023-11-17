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
@description: Blending operations for image inputs.
"""

import numpy as np
from PIL import Image, ImageChops
from ..util import *

# =============================================================================
# === EXPORT ===
# =============================================================================

__all__ = ["BlendNode"]

# =============================================================================
# === COMPOSITING ===
# =============================================================================

class BlendNode:
    """
    """
    OPS = {
        'LERP': Image.blend,
        'ADD': ImageChops.add,
        'MINIMUM': ImageChops.darker,
        'MAXIMUM': ImageChops.lighter,
        'MULTIPLY': ImageChops.multiply,
        'SOFT LIGHT': ImageChops.soft_light,
        'HARD LIGHT': ImageChops.hard_light,
        'OVERLAY': ImageChops.overlay,
        'SCREEN': ImageChops.screen,
        'SUBTRACT': ImageChops.subtract,
        'DIFFERENCE': ImageChops.difference,
        'LOGICAL AND': np.bitwise_and,
        'LOGICAL OR': np.bitwise_or,
        'LOGICAL XOR': np.bitwise_xor,
    }

    @classmethod
    def INPUT_TYPES(s):
        d = {"required": {
                    "imageA": ("IMAGE", ),
                    "imageB": ("IMAGE", ),
                    "alpha": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.05}),
                },
                "optional": {
                    "func": (list(BlendNode.OPS.keys()), {"default": "LERP"}),
                    "modeA": (["FIT", "CROP", "ASPECT"], {"default": "FIT"}),
                    "modeB": (["FIT", "CROP", "ASPECT"], {"default": "FIT"}),
            }}
        return deep_merge_dict(d, IT_WH, IT_INVERT)

    DESCRIPTION = "Takes 2 Image inputs and an apha and performs a linear blend (alpha) between both images based on the selected operations."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, imageA: torch.tensor, imageB: torch.tensor, alpha: float, func, modeA, modeB, width, height, invert):
        imageA = tensor2cv(imageA)
        imageA = SCALEFIT(imageA, width, height, modeA)
        h, w, _ = imageA.shape
        # print('BLEND', w, h)

        imageB = tensor2cv(imageB)
        imageB = SCALEFIT(imageB, width, height, modeB)
        h, w, _ = imageB.shape
        # print('BLEND', w, h)

        if (op := BlendNode.OPS.get(func, None)):
            alpha = min(max(alpha, 0.), 1.)
            if func == 'LERP':
                imageA = cv2pil(imageA)
                imageB = cv2pil(imageB)
                imageA = op(imageA, imageB, alpha)
                imageA = pil2cv(imageA)
            elif func.startswith("LOGICAL"):
                imageA = np.array(imageA)
                imageB = np.array(imageB)
                imageA = op(imageA, imageB)
                imageA = Image.fromarray(imageA)
                imageA = pil2cv(imageA)
            else:
                imageA = cv2pil(imageA)
                imageB = cv2pil(imageB)
                if func == 'MULTIPLY':
                    imageB = imageB.point(lambda i: 255 - int(i * alpha))
                else:
                    imageB = imageB.point(lambda i: int(i * alpha))
                imageA = pil2cv(op(imageA, imageB))

        # rebound to target width and height
        imageA = cv2.resize(imageA, (width, height))

        if invert:
            imageA = INVERT(imageA, invert)

        return (cv2tensor(imageA),)

NODE_CLASS_MAPPINGS = {
    "⚗️ Blend Images (jov)": BlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
