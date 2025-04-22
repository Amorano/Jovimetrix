""" Jovimetrix - Color """

from enum import Enum
from typing import List

import cv2
import torch

from comfy.utils import ProgressBar

from cozy_comfyui import \
    IMAGE_SIZE_MIN, \
    InputType, RGBAMaskType, EnumConvertType, TensorType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyBaseNode, CozyImageNode

from cozy_comfyui.image import \
    EnumImageType

from cozy_comfyui.image.convert import \
    image_mask, image_mask_add, tensor_to_cv, \
    cv_to_tensor, cv_to_tensor_full

from cozy_comfyui.image.misc import \
    image_stack

from .. import \
    Lexicon

from ..sup.image.color import \
    EnumCBDeficiency, EnumCBSimulator, EnumColorMap, EnumColorTheory, \
    color_lut_full, color_lut_match, color_lut_palette, \
    color_lut_tonal, color_lut_visualize, color_match_reinhard, \
    color_theory, color_blind, color_top_used, image_gradient_expand, \
    image_gradient_map

from ..sup.image.adjust import \
    EnumScaleMode, EnumInterpolation, \
    image_scalefit, image_invert

from ..sup.image.channel import \
    channel_solid

JOV_CATEGORY = "COLOR"

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumColorMatchMode(Enum):
    REINHARD = 30
    LUT = 10
    # HISTOGRAM = 20

class EnumColorMatchMap(Enum):
    USER_MAP = 0
    PRESET_MAP = 10

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ColorBlindNode(CozyImageNode):
    NAME = "COLOR BLIND (JOV) 👁‍🗨"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Simulate color blindness effects on images. You can select various types of color deficiencies, adjust the severity of the effect, and apply the simulation using different simulators. This node is ideal for accessibility testing and design adjustments, ensuring inclusivity in your visual content.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {}),
                "DEFICIENCY": (EnumCBDeficiency._member_names_, {
                    "default": EnumCBDeficiency.PROTAN.name,
                    "tooltip": "Type of color deficiency: Red (Protanopia), Green (Deuteranopia), Blue (Tritanopia)"
                }),
                "SIMULATOR": (EnumCBSimulator._member_names_, {
                    "default": EnumCBSimulator.AUTOSELECT.name,
                    "tooltip": "Solver to use when translating to new color space"
                }),
                "VAL": ("FLOAT", {
                    "default": 1, "min": 0, "max": 1, "step": 0.001,
                    "tooltip": "alpha blending"
                }),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        deficiency = parse_param(kw, "DEFICIENCY", EnumCBDeficiency, EnumCBDeficiency.PROTAN.name)
        simulator = parse_param(kw, "SIMULATOR", EnumCBSimulator, EnumCBSimulator.AUTOSELECT.name)
        severity = parse_param(kw, "VAL", EnumConvertType.FLOAT, 1)
        params = list(zip_longest_fill(pA, deficiency, simulator, severity))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, deficiency, simulator, severity) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor_to_cv(pA)
            pA = color_blind(pA, deficiency, simulator, severity)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class ColorMatchNode(CozyImageNode):
    NAME = "COLOR MATCH (JOV) 💞"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Adjust the color scheme of one image to match another with the Color Match Node. Choose from various color matching LUTs or Reinhard matching. You can specify a custom user color maps, the number of colors, and whether to flip or invert the images.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (COZY_TYPE_IMAGE, {}),
                Lexicon.PIXEL_B: (COZY_TYPE_IMAGE, {}),
                "MODE": (EnumColorMatchMode._member_names_, {
                    "default": EnumColorMatchMode.REINHARD.name,
                    "tooltip": "Match colors from an image or built-in (LUT), Histogram lookups or Reinhard method"
                }),
                "MAP": (EnumColorMatchMap._member_names_, {
                    "default": EnumColorMatchMap.USER_MAP.name,
                    "tooltip": "Custom image that will be transformed into a LUT or a built-in cv2 LUT"
                }),
                Lexicon.COLORMAP: (EnumColorMap._member_names_, {
                    "default": EnumColorMap.HSV.name
                }),
                "VAL": ("INT", {"default": 255, "min": 0, "max": 255, "tooltip":"The number of colors to use from the LUT during the remap. Will quantize the LUT range."}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False,
                                                "tooltip": "Invert the color match output"}),
                "MATTE": ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        colormatch_mode = parse_param(kw, "MODE", EnumColorMatchMode, EnumColorMatchMode.REINHARD.name)
        colormatch_map = parse_param(kw, f"MAP", EnumColorMatchMap, EnumColorMatchMap.USER_MAP.name)
        colormap = parse_param(kw, Lexicon.COLORMAP, EnumColorMap, EnumColorMap.HSV.name)
        num_colors = parse_param(kw, "VAL", EnumConvertType.INT, 255)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, pB, colormap, colormatch_mode, colormatch_map, num_colors, flip, invert, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, colormap, mode, cmap, num_colors, flip, invert, matte) in enumerate(params):
            if flip == True:
                pA, pB = pB, pA

            mask = None
            if pA is None:
                pA = channel_solid(chan=EnumImageType.BGR)
            else:
                pA = tensor_to_cv(pA)
                if pA.ndim == 3 and pA.shape[2] == 4:
                    mask = image_mask(pA)

            # h, w = pA.shape[:2]
            if pB is None:
                pB = channel_solid(chan=EnumImageType.BGR)
            else:
                pB = tensor_to_cv(pB)

            match mode:
                case EnumColorMatchMode.LUT:
                    if cmap == EnumColorMatchMap.PRESET_MAP:
                        pB = None
                    pA = color_lut_match(pA, colormap.value, pB, num_colors)

                case EnumColorMatchMode.REINHARD:
                    pA = color_match_reinhard(pA, pB)

            if invert == True:
                pA = image_invert(pA, 1)

            if mask is not None:
                pA = image_mask_add(pA, mask)

            images.append(cv_to_tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class ColorKMeansNode(CozyBaseNode):
    NAME = "COLOR MEANS (JOV) 〰️"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "JLUT", "IMAGE",)
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.PALETTE, Lexicon.GRADIENT, Lexicon.LUT, Lexicon.RGB, )
    OUTPUT_TOOLTIPS = (
        "Sequence of top-K colors. Count depends on value in `VAL`.",
        "Simple Tone palette based on result top-K colors. Width is taken from input.",
        "Gradient of top-K colors.",
        "Full 3D LUT of the image mapped to the resultant top-K colors chosen.",
        "Visualization of full 3D .cube LUT in JLUT output"
    )
    DESCRIPTION = """
The top-k colors ordered from most->least used as a strip, tonal palette and 3D LUT.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {}),
                "VAL": ("INT", {
                    "default": 12, "min": 1, "max": 255,
                    "tooltip":"The top K colors to select."
                }),
                Lexicon.SIZE: ("INT", {
                    "default": 32, "min": 1, "max": 256,
                    "tooltip":"Height of the tones in the strip. Width is based on input."
                }),
                "COUNT": ("INT", {
                    "default": 33, "min": 3, "max": 256,
                    "tooltip":"Number of nodes to use in interpolation of full LUT (256 is every pixel)."
                }),
                Lexicon.WH: ("VEC2INT", {
                    "default": (256, 256), "mij":IMAGE_SIZE_MIN,
                    "label": [Lexicon.W, Lexicon.H]
                }),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        kcolors = parse_param(kw, "VAL", EnumConvertType.INT, 12, 1, 255)
        lut_height = parse_param(kw, Lexicon.SIZE, EnumConvertType.INT, 32, 1, 256)
        nodes = parse_param(kw, "COUNT", EnumConvertType.INT, 33, 1, 255)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(256, 256)], IMAGE_SIZE_MIN)

        params = list(zip_longest_fill(pA, kcolors, nodes, lut_height, wihi))
        top_colors = []
        lut_tonal = []
        lut_full = []
        lut_visualized = []
        gradients = []
        pbar = ProgressBar(len(params) * sum(kcolors))
        for idx, (pA, kcolors, nodes, lut_height, wihi) in enumerate(params):
            if pA is None:
                pA = channel_solid(chan=EnumImageType.BGRA)

            pA = tensor_to_cv(pA)
            colors = color_top_used(pA, kcolors)

            # size down to 1px strip then expand to 256 for full gradient
            top_colors.extend([cv_to_tensor(channel_solid(*wihi, color=c)) for c in colors])
            lut_tonal.append(cv_to_tensor(color_lut_tonal(colors, width=pA.shape[1], height=lut_height)))
            full = color_lut_full(colors, nodes)
            lut_full.append(torch.from_numpy(full))
            lut_visualized.append(cv_to_tensor(color_lut_visualize(full, wihi[1])))
            gradient = image_gradient_expand(color_lut_palette(colors, 1))
            gradient = cv2.resize(gradient, wihi)
            gradients.append(cv_to_tensor(gradient))
            pbar.update_absolute(idx)

        return torch.stack(top_colors), torch.stack(lut_tonal), torch.stack(gradients), lut_full, torch.stack(lut_visualized),

class ColorTheoryNode(CozyBaseNode):
    NAME = "COLOR THEORY (JOV) 🛞"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (Lexicon.C1, Lexicon.C2, Lexicon.C3, Lexicon.C4, Lexicon.C5)
    SORT = 100
    DESCRIPTION = """
Generate a color harmony based on the selected scheme.

Supported schemes include complimentary, analogous, triadic, tetradic, and more.

Users can customize the angle of separation for color calculations, offering flexibility in color manipulation and exploration of different color palettes.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {}),
                "SCHEME": (EnumColorTheory._member_names_, {
                    "default": EnumColorTheory.COMPLIMENTARY.name
                }),
                "VAL": ("INT", {
                    "default": 45, "min": -90, "max": 90,
                    "tooltip": "Custom angle of separation to use when calculating colors"
                }),
                Lexicon.INVERT: ("BOOLEAN", {"default": False})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[List[TensorType], List[TensorType]]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        scheme = parse_param(kw, "SCHEME", EnumColorTheory, EnumColorTheory.COMPLIMENTARY.name)
        user = parse_param(kw, "VAL", EnumConvertType.INT, 0, -180, 180)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, scheme, user, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (img, target, user, invert) in enumerate(params):
            img = tensor_to_cv(img) if img is not None else channel_solid(chan=EnumImageType.BGRA)
            img = color_theory(img, user, target)
            if invert:
                img = (image_invert(s, 1) for s in img)
            images.append([cv_to_tensor(a) for a in img])
            pbar.update_absolute(idx)
        return image_stack(images)

class GradientMap(CozyImageNode):
    NAME = "GRADIENT MAP (JOV) 🇲🇺"
    CATEGORY = JOV_CATEGORY
    SORT = 550
    DESCRIPTION = """
Remaps an input image using a gradient lookup table (LUT).

The gradient image will be translated into a single row lookup table.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {
                    "tooltip":"Image to remap with gradient input"
                }),
                Lexicon.GRADIENT: (COZY_TYPE_IMAGE, {
                    "tooltip":f"Look up table (LUT) to remap the input image in `{Lexicon.PIXEL}`"
                }),
                Lexicon.FLIP: ("BOOLEAN", {
                    "default":False,
                    "tooltip":"Reverse the gradient from left-to-right "
                }),
                "MODE": (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name
                }),
                Lexicon.WH: ("VEC2INT", {
                    "default": (512, 512), "mij":IMAGE_SIZE_MIN,
                    "label": [Lexicon.W, Lexicon.H]
                }),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name
                }),
                "MATTE": ("VEC4INT", {
                    "default": (0, 0, 0, 255), "rgb": True
                })
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        gradient = parse_param(kw, Lexicon.GRADIENT, EnumConvertType.IMAGE, None)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, "MODE", EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], IMAGE_SIZE_MIN)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        images = []
        params = list(zip_longest_fill(pA, gradient, flip, mode, sample, wihi, matte))
        pbar = ProgressBar(len(params))
        for idx, (pA, gradient, flip, mode, sample, wihi, matte) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGR) if pA is None else tensor_to_cv(pA)
            mask = None
            if pA.ndim == 3 and pA.shape[2] == 4:
                mask = image_mask(pA)

            gradient = channel_solid(chan=EnumImageType.BGR) if gradient is None else tensor_to_cv(gradient)
            pA = image_gradient_map(pA, gradient)
            if mode != EnumScaleMode.MATTE:
                w, h = wihi
                pA = image_scalefit(pA, w, h, mode, sample)
            if mask is not None:
                pA = image_mask_add(pA, mask)
            images.append(cv_to_tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return image_stack(images)
