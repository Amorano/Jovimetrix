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
@description: Constant color.
"""

from PIL import Image

from .. import IT_COLOR, IT_WH, deep_merge_dict
from ..util import JovimetrixBaseNode, pil2tensor

__all__ = ["ConstantNode"]

# =============================================================================
class ConstantNode(JovimetrixBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        return deep_merge_dict(IT_WH, IT_COLOR)

    DESCRIPTION = ""
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/Image"

    def run(self, width, height, R, G, B):
        image = Image.new("RGB", (width, height), (int(R * 255.), int(G * 255.), int(B * 255.)) )
        return (pil2tensor(image),)

NODE_CLASS_MAPPINGS = {
    "🟪 Constant Image (jov)": ConstantNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
