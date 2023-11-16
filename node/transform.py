"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix
"""

import torch
import numpy as np

from ..util import *

# =============================================================================
# === EXPORT ===
# =============================================================================

__all__ = ["TransformNode", "InvertNode", "MirrorNode", "HSVNode", "AdjustmentNode"]

# =============================================================================
# === SINGLE IMAGE MANIPUALTION ===
# =============================================================================
# offset: coords should be in range -0.5..0.5
class TransformNode:
    @classmethod
    def INPUT_TYPES(s):
        return deep_merge_dict(IT_IMAGE, IT_TRS, IT_EDGE, IT_WH, IT_WHMODE)

    DESCRIPTION = "Translate, Rotate, Scale, Tile and Invert an Image. All options allow for CROP or WRAPing of the edges."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, offsetX, offsetY, angle, sizeX, sizeY, edge, width, height, mode):
        image = tensor2cv(image)
        image = TRANSFORM(image, offsetX, offsetY, angle, sizeX, sizeY, edge, width, height, mode)
        return (cv2tensor(image),)

class TileNode:
    @classmethod
    def INPUT_TYPES(s):
        return deep_merge_dict(IT_IMAGE, IT_TILE)

    DESCRIPTION = "Tile an Image with optional crop to original image size."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, tileX, tileY):
        image = tensor2cv(image)
        height, width, _ = image.shape
        image = EDGEWRAP(image, tileX, tileY)
        # rebound to target width and height
        image = cv2.resize(image, (width, height))
        return (cv2tensor(image),)

#
class InvertNode:
    @classmethod
    def INPUT_TYPES(s):
        d = {"required": {
            "alpha": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.05}),
        }}
        return deep_merge_dict(IT_IMAGE, d)

    DESCRIPTION = "Alpha blend an Image's inverted version. with the original."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, alpha):
        image = tensor2cv(image)
        image = INVERT(image, alpha)
        return (cv2tensor(image),)

#
class MirrorNode:
    @classmethod
    def INPUT_TYPES(s):
        d = {
            "required": {
                "x": ("FLOAT", {"default": 0.5, "min": 0., "max": 1., "step": 0.05}),
                "y": ("FLOAT", {"default": 0.5, "min": 0., "max": 1., "step": 0.05}),
                "mode": (["X", "Y", "XY", "YX"], {"default": "X"}),
            },
        }
        return deep_merge_dict(IT_IMAGE, d, IT_INVERT)

    DESCRIPTION = "Flip an Image across the X axis, the Y Axis or both, with independant centers."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, x, y, mode, invert):
        image = tensor2cv(image)
        while (len(mode) > 0):
            axis, mode = mode[0], mode[1:]
            if axis == 'X':
                image = MIRROR(image, x, 1, invert=invert)
            else:
                image = MIRROR(image, 1 - y, 0, invert=invert)
        return (cv2tensor(image),)

#
class HSVNode:
    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "optional": {
                "hue": ("FLOAT",{"default": 0.5, "min": 0., "max": 1., "step": 0.02},),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.02}, ),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.02}, ),
            }
        }
        return deep_merge_dict(IT_IMAGE, d)

    DESCRIPTION = "Tweak the Hue, Saturation and Value for an Image."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, hue, saturation, value):
        if hue != 0.5 or saturation != 1. or value != 1.:
            image = HSV(image, hue, saturation, value)
            image = torch.clamp(image, 0.0, 1.0)
        return (image,)

class LumenNode:

    OPS = {
        'CONTRAST': CONTRAST,
        'GAMMA': GAMMA,
        'EXPOSURE': EXPOSURE,
    }

    @classmethod
    def INPUT_TYPES(cls):
        d = {
            "optional": {
                "op": (["CONTRAST", "GAMMA", "EXPOSURE"], ),
                "adjust": ("FLOAT",{"default": 1., "min": 0., "max": 1., "step": 0.02},),
            }
        }
        return deep_merge_dict(IT_IMAGE, d)

    DESCRIPTION = "Contrast, Gamma and Exposure controls."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, op, adjust):
        image = LumenNode.OPS[op](image, adjust)
        return (image,)

NODE_CLASS_MAPPINGS = {
    "🌱 Transform Image (jov)": TransformNode,
    "🔳 Tile Image (jov)": TileNode,
    "🎭 Invert Image (jov)": InvertNode,
    "🔰 Mirror Image (jov)": MirrorNode,
    "🌈 HSV Image (jov)": HSVNode,
    "🔧 Adjust Image (jov)": LumenNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
