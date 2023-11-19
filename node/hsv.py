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

from .. import IT_IMAGE, deep_merge_dict
from ..util import JovimetrixBaseNode, HSV, tensor2cv, cv2tensor

__all__ = ["HSVNode"]

# =============================================================================
class HSVNode(JovimetrixBaseNode):
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

    def run(self, image, hue, saturation, value):
        image = tensor2cv(image)
        if hue != 0.5 or saturation != 1. or value != 1.:
            image = HSV(image, hue, saturation, value)
        return (cv2tensor(image),)

NODE_CLASS_MAPPINGS = {
    "🌈 HSV (jov)": HSVNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
