"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix
"""

from PIL import ImageFilter

from ..util import *

# =============================================================================
# === EXPORT ===
# =============================================================================

__all__ = ["FilterNode"]

# =============================================================================
# === FILTERING ===
# =============================================================================
class FilterNode:
    OPS = {
        'BLUR': ImageFilter.GaussianBlur,
        'SHARPEN': ImageFilter.UnsharpMask,
    }

    OPS_PRE = {
        # PREDEFINED
        'EMBOSS': ImageFilter.EMBOSS,
        'FIND_EDGES': ImageFilter.FIND_EDGES,
    }
    @classmethod
    def INPUT_TYPES(s):
        ops = list(FilterNode.OPS.keys()) + list(FilterNode.OPS_PRE.keys())
        d = {"required": {
                    "func": (ops, {"default": "BLUR"}),
            },
            "optional": {
                "radius": ("INT", {"default": 1, "min": 0, "step": 1}),
        }}
        return deep_merge_dict(IT_IMAGE, d)

    DESCRIPTION = "A single node with multiple operations."
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "run"

    def run(self, image, func, radius):
        image = tensor2pil(image)

        if (op := FilterNode.OPS.get(func, None)):
           image = image.filter(op(radius))

        elif (op := FilterNode.OPS_PRE.get(func, None)):
            image = image.filter(op())

        return (pil2tensor(image),)

NODE_CLASS_MAPPINGS = {
    "🕸️ Filter Image (jov)": FilterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
