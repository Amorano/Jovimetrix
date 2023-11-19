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
@description: HSV Image inputs.
"""

from .. import IT_WH, IT_PIXEL2, IT_WHMODE, deep_merge_dict
from ..util import JovimetrixBaseNode, EXTEND, SCALEFIT, tensor2cv, cv2mask, cv2tensor

__all__ = ["ExtendNode"]

# =============================================================================
class ExtendNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        d = {"required": {
                "axis": (["HORIZONTAL", "VERTICAL"], {"default": "HORIZONTAL"}),
            },
            "optional": {
                "flip": ("BOOLEAN", {"default": False}),
            },
        }
        return deep_merge_dict(IT_PIXEL2, d, IT_WH, IT_WHMODE)

    DESCRIPTION = "Contrast, Gamma and Exposure controls for images."

    def run(self, pixelA, pixelB, axis, flip, width, height, mode):
        pixelA = tensor2cv(pixelA)
        pixelB = tensor2cv(pixelB)
        pixelA = EXTEND(pixelA, pixelB, axis, flip)
        if mode != "NONE":
            pixelA = SCALEFIT(pixelA, width, height, mode)
        return (cv2tensor(pixelA), cv2mask(pixelA), )

NODE_CLASS_MAPPINGS = {
    "🎇 Expand (jov)": ExtendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
