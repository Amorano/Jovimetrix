"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix
"""

import numpy as np
from PIL import Image, ImageFilter

from ..util import *

# =============================================================================
# === EXPORT ===
# =============================================================================

__all__ = ["MappingNode"]

# =============================================================================
# === MAPPING ===
# =============================================================================
class MappingNode:
    @classmethod
    def INPUT_TYPES(s):
        d = {"required": {
                "proj": (["SPHERICAL", "CYLINDRICAL"], {"default": "SPHERICAL"}),
            }}
        return deep_merge_dict(IT_IMAGE, d, IT_WH)

    DESCRIPTION = ""
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHAPE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, proj, width, height):
        image = tensor2pil(image)

        source_width, source_height = image.size
        target_image = Image.new("RGB", (width, height))
        for y_target in range(height):
            for x_target in range(width):
                x_source = int((x_target / width) * source_width)

                if proj == "SPHERICAL":
                    x_source %= source_width
                y_source = int(y_target / height * source_height)
                px = image.getpixel((x_source, y_source))

                target_image.putpixel((x_target, y_target), px)
        return (pil2tensor(target_image),)

NODE_CLASS_MAPPINGS = {
    "🗺️ Remap Image (jov)": MappingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
