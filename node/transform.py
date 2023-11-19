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
@description: Transform inputs
"""

from .. import IT_TRS, IT_WH, IT_PIXELS, IT_EDGE, IT_WHMODE, deep_merge_dict
from ..util import JovimetrixBaseNode, TRANSFORM, tensor2cv, cv2mask, cv2tensor

__all__ = ["TransformNode"]

# =============================================================================
class TransformNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        return deep_merge_dict(IT_PIXELS, IT_TRS, IT_EDGE, IT_WH, IT_WHMODE)

    DESCRIPTION = "Translate, Rotate, Scale, Tile and Invert an input. All options allow for CROP or WRAPing of the edges."

    def run(self, pixels, offsetX, offsetY, angle, sizeX, sizeY, edge, width, height, mode):
        pixels = tensor2cv(pixels)
        pixels = TRANSFORM(pixels, offsetX, offsetY, angle, sizeX, sizeY, edge, width, height, mode)
        return (cv2tensor(pixels), cv2mask(pixels), )

NODE_CLASS_MAPPINGS = {
    "🌱 Transform Image (jov)": TransformNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
