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
@description: Invert inputs.
"""

from .. import IT_PIXELS, deep_merge_dict
from ..util import JovimetrixBaseNode, INVERT, tensor2cv, cv2mask, cv2tensor

__all__ = ["InvertNode"]

# =============================================================================
class InvertNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        d = {"required": {
            "alpha": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.05}),
        }}
        return deep_merge_dict(IT_PIXELS, d)

    DESCRIPTION = "Alpha blend an input's inverted version with the original."

    def run(self, pixels, alpha):
        pixels = tensor2cv(pixels)
        pixels = INVERT(pixels, alpha)
        return (cv2tensor(pixels), cv2mask(pixels), )

NODE_CLASS_MAPPINGS = {
    "🎭 Invert (jov)": InvertNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
