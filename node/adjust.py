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
@description: Adjustments (Gamma, Exposure, Inversion, Emoss, Edges, etc..).
"""

from PIL import ImageFilter
from .. import deep_merge_dict, IT_PIXELS
from ..util import JovimetrixBaseNode, INVERT, CONTRAST, GAMMA, EXPOSURE, \
    tensor2pil, tensor2cv, cv2mask, cv2tensor, cv2pil

__all__ = ["AdjustNode"]

# =============================================================================
class AdjustNode(JovimetrixBaseNode):
    OPS = {
        'BLUR': ImageFilter.GaussianBlur,
        'SHARPEN': ImageFilter.UnsharpMask,
    }

    OPS_PRE = {
        # PREDEFINED
        'EMBOSS': ImageFilter.EMBOSS,
        'FIND_EDGES': ImageFilter.FIND_EDGES,
    }

    OPS_CV2 = {
        'CONTRAST': CONTRAST,
        'GAMMA': GAMMA,
        'EXPOSURE': EXPOSURE,
        'INVERT': None,
    }

    @classmethod
    def INPUT_TYPES(s):
        ops = list(AdjustNode.OPS.keys()) + list(AdjustNode.OPS_PRE.keys()) + list(AdjustNode.OPS_CV2.keys())
        d = {"required": {
                "func": (ops, {"default": "BLUR"}),
            },
            "optional": {
                "radius": ("INT", {"default": 1, "min": 0, "step": 1}),
                "alpha": ("FLOAT",{"default": 1., "min": 0., "max": 1., "step": 0.02},),
        }}
        return deep_merge_dict(IT_PIXELS, d)

    DESCRIPTION = "A single node with multiple operations."

    def run(self, pixels, func, radius, alpha):
        if (op := AdjustNode.OPS.get(func, None)):
           pixels = tensor2pil(pixels)
           pixels = pixels.filter(op(radius))
        elif (op := AdjustNode.OPS_PRE.get(func, None)):
            pixels = tensor2pil(pixels)
            pixels = pixels.filter(op())
        elif (op := AdjustNode.OPS_CV2.get(func, None)):
            pixels = tensor2cv(pixels)
            if op == 'INVERT':
                pixels = INVERT(pixels, alpha)
            else:
                pixels = op(pixels, alpha)
            pixels = cv2pil(pixels)
        return (cv2tensor(pixels), cv2mask(pixels), )

NODE_CLASS_MAPPINGS = {
    "🕸️ Adjust (jov)": AdjustNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
