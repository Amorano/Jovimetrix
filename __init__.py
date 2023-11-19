"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix
"""

import os
import importlib
import traceback
import logging

logger = logging.getLogger(__package__)
logger.setLevel(logging.INFO)

# =============================================================================
# === UTILITY ===
# =============================================================================

def load_nodes():
    error_messages = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    # folder where the nodes are stored
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(root, "node")

    # get everything
    module = [os.path.splitext(m)[0] for m in os.listdir(root)]
    # specific order for these
    order = ["constant", "geometrix", "transform", "mirror", "tile", "hsv", "adjust", "filtering", "blend", "mapping"]
    for m in order:
        try:
            module.remove(m)
        except ValueError as _:
            pass
    # toss the rest on the heap
    order.extend(module)

    for module_name in order:
        try:
            module = importlib.import_module(
                f".node.{module_name}", package = __package__
            )
        except:
            error_messages.append(f"❗❗IMPORT❗❗ '{module_name}'\n{traceback.format_exc()}")
            continue

        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))

        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))

        logger.debug(f"✔️ '{module_name}' (JOVIMETRIX)")

    if len(error_messages) > 0:
        logger.warning(
            f"⚠️ failed to load:\n\n"
            + "\n".join(error_messages)
        )

    return node_class_mappings, node_display_name_mappings

# =============================================================================
# === GLOBAL ===
# =============================================================================

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("🖖")

##
#
#

def deep_merge_dict(*dicts: dict) -> dict:
    """
    Deep merge multiple dictionaries recursively.
    """
    def _deep_merge(d1, d2):
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            return d2

        merged_dict = d1.copy()

        for key in d2:
            if key in merged_dict:
                if isinstance(merged_dict[key], dict) and isinstance(d2[key], dict):
                    merged_dict[key] = _deep_merge(merged_dict[key], d2[key])
                elif isinstance(merged_dict[key], list) and isinstance(d2[key], list):
                    merged_dict[key].extend(d2[key])
                else:
                    merged_dict[key] = d2[key]
            else:
                merged_dict[key] = d2[key]
        return merged_dict

    merged = {}
    for d in dicts:
        merged = _deep_merge(merged, d)
    return merged

##
##

IT_IMAGE = {
    "required": {
        "image": ("IMAGE", ),
    }
}

IT_MASK = {
    "required": {
        "mask": ("MASK", ),
    }
}

IT_PIXELS = {
    "required": {
        "pixels": (any_typ, {"default": None}),
    }
}

IT_PIXEL2 = {
    "required": {
        "pixelA": (any_typ, {"default": None}),
        "pixelB": (any_typ, {"default": None}),
    }
}

IT_WH = {
    "required":{},
    "optional": {
        "width": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 64}),
        "height": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 64}),
    }
}

IT_WHMODE = {
    "required":{},
    "optional": {
        "mode": (["NONE", "FIT", "CROP", "ASPECT"], {"default": "NONE"}),
    }
}

IT_TRANS = {
    "required":{},
    "optional": {
        "offsetX": ("FLOAT", {"default": 0., "min": -1., "max": 1., "step": 0.05}),
        "offsetY": ("FLOAT", {"default": 0., "min": -1., "max": 1., "step": 0.05, "display": "number"}),
    }
}

IT_ROT = {
    "required":{},
    "optional": {
        "angle": ("FLOAT", {"default": 0., "min": -180., "max": 180., "step": 5., "display": "number"}),
    }
}

IT_SCALE = {
    "required":{},
    "optional": {
        "sizeX": ("FLOAT", {"default": 1., "min": 0.01, "max": 2., "step": 0.05}),
        "sizeY": ("FLOAT", {"default": 1., "min": 0.01, "max": 2., "step": 0.05}),
    }
}

IT_TILE = {
    "required":{},
    "optional": {
        "tileX": ("INT", {"default": 1, "min": 0, "step": 1, "display": "number"}),
        "tileY": ("INT", {"default": 1, "min": 0, "step": 1}),
    }
}

IT_EDGE = {
    "required":{},
    "optional": {
        "edge": (["CLIP", "WRAP", "WRAPX", "WRAPY"], {"default": "CLIP"}),
    }
}

IT_INVERT = {
    "required":{},
    "optional": {
        "invert": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": 0.25}),
    }
}

IT_COLOR = {
    "required":{},
    "optional": {
        "R": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.1}),
        "G": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.1}),
        "B": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 0.1}),
    }
}

# Translate, Rotate, Scale Params
IT_TRS = deep_merge_dict(IT_TRANS, IT_ROT, IT_SCALE)

IT_WHFULL = deep_merge_dict(IT_WH, IT_WHMODE, IT_INVERT)


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()
