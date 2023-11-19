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
@description: Tile inputs.
"""

import cv2
from .. import IT_PIXELS, IT_TILE, deep_merge_dict
from ..util import JovimetrixBaseNode, EDGEWRAP, tensor2cv, cv2mask, cv2tensor

__all__ = ["TileNode"]

# =============================================================================
class TileNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        return deep_merge_dict(IT_PIXELS, IT_TILE)

    DESCRIPTION = "Tile an Image with optional crop to original image size."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"

    def run(self, pixels, tileX, tileY):
        pixels = tensor2cv(pixels)
        height, width, _ = pixels.shape
        pixels = EDGEWRAP(pixels, tileX, tileY)
        # rebound to target width and height
        pixels = cv2.resize(pixels, (width, height))
        return (cv2tensor(pixels), cv2mask(pixels), )

NODE_CLASS_MAPPINGS = {
    "🔳 Tile (jov)": TileNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
