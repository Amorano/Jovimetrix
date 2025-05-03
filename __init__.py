"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

              Animation, Image Compositing & Procedural Creation

@title: Jovimetrix
@author: Alexander G. Morano
@category: Compositing
@reference: https://github.com/Amorano/Jovimetrix
@tags: adjust, animate, compose, compositing, composition, device, flow, video,
mask, shape, animation, logic
@description: Animation via tick. Parameter manipulation with wave generator.
Unary and Binary math support. Value convert int/float/bool, VectorN and Image,
Mask types. Shape mask generator. Stack images, do channel ops, split, merge
and randomize arrays and batches. Load images & video from anywhere. Dynamic
bus routing. Save output anywhere! Flatten, crop, transform; check
colorblindness or linear interpolate values.
@node list:
    TickNode, TickSimpleNode, WaveGeneratorNode
    BitSplitNode, ComparisonNode, LerpNode, OPUnaryNode, OPBinaryNode, StringerNode, SwizzleNode,
    ColorBlindNode, ColorMatchNode, ColorKMeansNode, ColorTheoryNode, GradientMapNode,
    AdjustNode, BlendNode, FilterMaskNode, PixelMergeNode, PixelSplitNode, PixelSwapNode, ThresholdNode,
    ConstantNode, ShapeNode, TextNode,
    CropNode, FlattenNode, StackNode, TransformNode,

    ArrayNode, QueueNode, QueueTooNode,
    AkashicNode, GraphNode, ImageInfoNode,
    DelayNode, ExportNode, RouteNode, SaveOutputNode

    ValueNode, Vector2Node, Vector3Node, Vector4Node,
"""

__author__ = "Alexander G. Morano"
__email__ = "amorano@gmail.com"

from pathlib import Path

from cozy_comfyui import \
    logger

from cozy_comfyui.node import \
    loader

JOV_DOCKERENV = False
try:
    with open('/proc/1/cgroup', 'rt') as f:
        content = f.read()
        JOV_DOCKERENV = any(x in content for x in ['docker', 'kubepods', 'containerd'])
except FileNotFoundError:
    pass

if JOV_DOCKERENV:
    logger.info("RUNNING IN A DOCKER")

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

PACKAGE = "JOVIMETRIX"
WEB_DIRECTORY = "./web"
ROOT = Path(__file__).resolve().parent
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = loader(ROOT,
                                                         PACKAGE,
                                                         "core",
                                                         f"{PACKAGE} 🔺🟩🔵",
                                                         False)
