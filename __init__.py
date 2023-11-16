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

    order = ["geometrix", "transform", "filtering", "blend", "mapping", "wip"]
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

IT_IMAGE = {
    "required": {
        "image": ("IMAGE", ),
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
        "mode": (["FIT", "CROP", "ASPECT"], {"default": "FIT"}),
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

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()
