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
@description: Mirror inputs.
"""

from .. import IT_PIXELS, IT_INVERT, deep_merge_dict
from ..util import JovimetrixBaseNode, MIRROR, tensor2cv, cv2mask, cv2tensor

__all__ = ["MirrorNode"]

# =============================================================================
class MirrorNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        d = {"required": {
                "x": ("FLOAT", {"default": 0.5, "min": 0., "max": 1., "step": 0.05}),
                "y": ("FLOAT", {"default": 0.5, "min": 0., "max": 1., "step": 0.05}),
                "mode": (["X", "Y", "XY", "YX"], {"default": "X"}),
            },
        }
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    DESCRIPTION = "Flip an input across the X axis, the Y Axis or both, with independant centers."

    def run(self, pixels, x, y, mode, invert):
        pixels = tensor2cv(pixels)
        while (len(mode) > 0):
            axis, mode = mode[0], mode[1:]
            px = [y, x][axis == 'X']
            pixels = MIRROR(pixels, px, axis == 'X', invert=invert)
        return (cv2tensor(pixels), cv2mask(pixels), )

NODE_CLASS_MAPPINGS = {
    "🔰 Mirror (jov)": MirrorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
