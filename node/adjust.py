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
@description: Adjust Gamma, Contrast or Exposure on input.
"""

from .. import deep_merge_dict, IT_MASK
from ..util import JovimetrixBaseNode, CONTRAST, GAMMA, EXPOSURE, cv2mask, cv2tensor

__all__ = ["AdjustNode"]
# =============================================================================
class AdjustNode(JovimetrixBaseNode):
    OPS = {
        'CONTRAST': CONTRAST,
        'GAMMA': GAMMA,
        'EXPOSURE': EXPOSURE,
    }

    @classmethod
    def INPUT_TYPES(s):
        d = {
            "optional": {
                "op": (["CONTRAST", "GAMMA", "EXPOSURE"], ),
                "adjust": ("FLOAT",{"default": 1., "min": 0., "max": 1., "step": 0.02},),
            }
        }
        return deep_merge_dict(IT_MASK, d)

    DESCRIPTION = "Contrast, Gamma and Exposure controls for an input."

    def run(self, pixels, op, adjust):
        pixels = AdjustNode.OPS[op](pixels, adjust)
        return (cv2tensor(pixels), cv2mask(pixels), )

NODE_CLASS_MAPPINGS = {
    "🔧 Adjust (jov)": AdjustNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
