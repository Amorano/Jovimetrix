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

import torch
import numpy as np
from PIL import Image, ImageChops
from .. import deep_merge_dict, IT_WHFULL
from ..util import cv2pil, pil2cv, cv2tensor, tensor2cv, JovimetrixBaseNode, SCALEFIT, INVERT

__all__ = ["BlendNode", "BlendMaskNode"]

# =============================================================================
# === COMPOSITING ===
# =============================================================================

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
_OPS = list(OPS.keys())

def blend(maskA, maskB, alpha, func):
    if (op := OPS.get(func, None)):
        alpha = min(max(alpha, 0.), 1.)
        if func == 'LERP':
            maskA = cv2pil(maskA)
            maskB = cv2pil(maskB)
            maskA = op(maskA, maskB, alpha)
            maskA = pil2cv(maskA)
        elif func.startswith("LOGICAL"):
            maskA = np.array(maskA)
            maskB = np.array(maskB)
            maskA = op(maskA, maskB)
            maskA = Image.fromarray(maskA)
            maskA = pil2cv(maskA)
        else:
            maskA = cv2pil(maskA)
            maskB = cv2pil(maskB)
            if func == 'MULTIPLY':
                maskB = maskB.point(lambda i: 255 - int(i * alpha))
            else:
                maskB = maskB.point(lambda i: int(i * alpha))
            maskA = pil2cv(op(maskA, maskB))
    return maskA

class BlendNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        d = {"required": {
                    "imageA": ("IMAGE", ),
                    "imageB": ("IMAGE", ),
                    "alpha": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.01}),
                },
                "optional": {
                    "func": (_OPS, {"default": "LERP"}),
                    "modeA": (["FIT", "CROP", "ASPECT"], {"default": "FIT"}),
                    "modeB": (["FIT", "CROP", "ASPECT"], {"default": "FIT"}),
            }}
        return deep_merge_dict(d, IT_WHFULL)

    DESCRIPTION = "Takes 2 Image inputs and an apha and performs a linear blend (alpha) between both images based on the selected operations."

    def run(self, imageA: torch.tensor, imageB: torch.tensor, alpha: float, func, modeA, modeB, width, height, mode, invert):
        imageA = tensor2cv(imageA)
        imageB = tensor2cv(imageB)

        imageA = SCALEFIT(imageA, width, height, modeA)
        h, w, _ = imageA.shape
        print('BLEND', func, w, h)

        imageB = SCALEFIT(imageB, width, height, modeB)
        h, w, _ = imageB.shape
        print('BLEND', w, h)

        imageA = blend(imageA, imageB, alpha, func)

        if invert:
            imageA = INVERT(imageA, invert)

        imageA = SCALEFIT(imageA, width, height, mode)
        return (cv2tensor(imageA),)

NODE_CLASS_MAPPINGS = {
    "⚗️ Blend Images (jov)": BlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
