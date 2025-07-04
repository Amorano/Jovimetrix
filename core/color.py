""" Jovimetrix - Color """

from enum import Enum

import cv2
import torch

from comfy.utils import ProgressBar

from cozy_comfyui import \
    IMAGE_SIZE_MIN, \
    InputType, RGBAMaskType, EnumConvertType, TensorType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyBaseNode, CozyImageNode

from cozy_comfyui.image.adjust import \
    image_invert

from cozy_comfyui.image.color import \
    EnumCBDeficiency, EnumCBSimulator, EnumColorMap, EnumColorTheory, \
    color_lut_full, color_lut_match, color_lut_palette, \
    color_lut_tonal, color_lut_visualize, color_match_reinhard, \
    color_theory, color_blind, color_top_used, image_gradient_expand, \
    image_gradient_map

from cozy_comfyui.image.channel import \
    channel_solid

from cozy_comfyui.image.compose import \
    EnumScaleMode, EnumInterpolation, \
    image_scalefit

from cozy_comfyui.image.convert import \
    tensor_to_cv, cv_to_tensor, cv_to_tensor_full, image_mask, image_mask_add

from cozy_comfyui.image.misc import \
    image_stack

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

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
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.DEFICIENCY: (EnumCBDeficiency._member_names_, {
                    "default": EnumCBDeficiency.PROTAN.name,}),
                Lexicon.SOLVER: (EnumCBSimulator._member_names_, {
                    "default": EnumCBSimulator.AUTOSELECT.name,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        deficiency = parse_param(kw, Lexicon.DEFICIENCY, EnumCBDeficiency, EnumCBDeficiency.PROTAN.name)
        simulator = parse_param(kw, Lexicon.SOLVER, EnumCBSimulator, EnumCBSimulator.AUTOSELECT.name)
        severity = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 1)
        params = list(zip_longest_fill(pA, deficiency, simulator, severity))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, deficiency, simulator, severity) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
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
                Lexicon.IMAGE_SOURCE: (COZY_TYPE_IMAGE, {}),
                Lexicon.IMAGE_TARGET: (COZY_TYPE_IMAGE, {}),
                Lexicon.MODE: (EnumColorMatchMode._member_names_, {
                    "default": EnumColorMatchMode.REINHARD.name,
                    "tooltip": "Match colors from an image or built-in (LUT), Histogram lookups or Reinhard method"}),
                Lexicon.MAP: (EnumColorMatchMap._member_names_, {
                    "default": EnumColorMatchMap.USER_MAP.name, }),
                Lexicon.COLORMAP: (EnumColorMap._member_names_, {
                    "default": EnumColorMap.HSV.name,}),
                Lexicon.VALUE: ("INT", {
                    "default": 255, "min": 0, "max": 255,
                    "tooltip":"The number of colors to use from the LUT during the remap. Will quantize the LUT range."}),
                Lexicon.SWAP: ("BOOLEAN", {
                    "default": False,}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE_SOURCE, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.IMAGE_TARGET, EnumConvertType.IMAGE, None)
        mode = parse_param(kw, Lexicon.MODE, EnumColorMatchMode, EnumColorMatchMode.REINHARD.name)
        cmap = parse_param(kw, Lexicon.MAP, EnumColorMatchMap, EnumColorMatchMap.USER_MAP.name)
        colormap = parse_param(kw, Lexicon.COLORMAP, EnumColorMap, EnumColorMap.HSV.name)
        num_colors = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 255)
        swap = parse_param(kw, Lexicon.SWAP, EnumConvertType.BOOLEAN, False)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4, (0, 0, 0, 255), 0, 255)
        params = list(zip_longest_fill(pA, pB, mode, cmap, colormap, num_colors, swap, invert, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, mode, cmap, colormap, num_colors, swap, invert, matte) in enumerate(params):
            if swap == True:
                pA, pB = pB, pA

            mask = None
            if pA is None:
                pA = channel_solid()
            else:
                pA = tensor_to_cv(pA)
                if pA.ndim == 3 and pA.shape[2] == 4:
                    mask = image_mask(pA)

            # h, w = pA.shape[:2]
            if pB is None:
                pB = channel_solid()
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
    RETURN_NAMES = ("IMAGE", "PALETTE", "GRADIENT", "LUT", "RGB", )
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
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.VALUE: ("INT", {
                    "default": 12, "min": 1, "max": 255,
                    "tooltip": "The top K colors to select"}),
                Lexicon.SIZE: ("INT", {
                    "default": 32, "min": 1, "max": 256,
                    "tooltip": "Height of the tones in the strip. Width is based on input"}),
                Lexicon.COUNT: ("INT", {
                    "default": 33, "min": 1, "max": 255,
                    "tooltip": "Number of nodes to use in interpolation of full LUT (256 is every pixel)"}),
                Lexicon.WH: ("VEC2", {
                    "default": (256, 256), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"]
                }),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        kcolors = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 12, 1, 255)
        lut_height = parse_param(kw, Lexicon.SIZE, EnumConvertType.INT, 32, 1, 256)
        nodes = parse_param(kw, Lexicon.COUNT, EnumConvertType.INT, 33, 1, 255)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (256, 256), IMAGE_SIZE_MIN)

        params = list(zip_longest_fill(pA, kcolors, nodes, lut_height, wihi))
        top_colors = []
        lut_tonal = []
        lut_full = []
        lut_visualized = []
        gradients = []
        pbar = ProgressBar(len(params) * sum(kcolors))
        for idx, (pA, kcolors, nodes, lut_height, wihi) in enumerate(params):
            if pA is None:
                pA = channel_solid()

            pA = tensor_to_cv(pA)
            colors = color_top_used(pA, kcolors)

            # size down to 1px strip then expand to 256 for full gradient
            top_colors.extend([cv_to_tensor(channel_solid(*wihi, color=c)) for c in colors])
            lut = color_lut_tonal(colors, width=pA.shape[1], height=lut_height)
            lut_tonal.append(cv_to_tensor(lut))
            full = color_lut_full(colors, nodes)
            lut_full.append(torch.from_numpy(full))
            lut = color_lut_visualize(full, wihi[1])
            lut_visualized.append(cv_to_tensor(lut))
            palette = color_lut_palette(colors, 1)
            gradient = image_gradient_expand(palette)
            gradient = cv2.resize(gradient, wihi)
            gradients.append(cv_to_tensor(gradient))
            pbar.update_absolute(idx)

        return torch.stack(top_colors), torch.stack(lut_tonal), torch.stack(gradients), lut_full, torch.stack(lut_visualized),

class ColorTheoryNode(CozyBaseNode):
    NAME = "COLOR THEORY (JOV) 🛞"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("C1", "C2", "C3", "C4", "C5")
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
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.SCHEME: (EnumColorTheory._member_names_, {
                    "default": EnumColorTheory.COMPLIMENTARY.name}),
                Lexicon.VALUE: ("INT", {
                    "default": 45, "min": -90, "max": 90,
                    "tooltip": "Custom angle of separation to use when calculating colors"}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[list[TensorType], list[TensorType]]:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        scheme = parse_param(kw, Lexicon.SCHEME, EnumColorTheory, EnumColorTheory.COMPLIMENTARY.name)
        value = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 45, -90, 90)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, scheme, value, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (img, scheme, value, invert) in enumerate(params):
            img = channel_solid() if img is None else tensor_to_cv(img)
            img = color_theory(img, value, scheme)
            if invert:
                img = (image_invert(s, 1) for s in img)
            images.append([cv_to_tensor(a) for a in img])
            pbar.update_absolute(idx)
        return image_stack(images)

class GradientMapNode(CozyImageNode):
    NAME = "GRADIENT MAP (JOV) 🇲🇺"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Remaps an input image using a gradient lookup table (LUT).

The gradient image will be translated into a single row lookup table.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {
                    "tooltip": "Image to remap with gradient input"}),
                Lexicon.GRADIENT: (COZY_TYPE_IMAGE, {
                    "tooltip": f"Look up table (LUT) to remap the input image in `{"IMAGE"}`"}),
                Lexicon.REVERSE: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the gradient from left-to-right"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name,}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"] }),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        gradient = parse_param(kw, Lexicon.GRADIENT, EnumConvertType.IMAGE, None)
        reverse = parse_param(kw, Lexicon.REVERSE, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        images = []
        params = list(zip_longest_fill(pA, gradient, reverse, mode, sample, wihi, matte))
        pbar = ProgressBar(len(params))
        for idx, (pA, gradient, reverse, mode, sample, wihi, matte) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            mask = None
            if pA.ndim == 3 and pA.shape[2] == 4:
                mask = image_mask(pA)

            gradient = channel_solid() if gradient is None else tensor_to_cv(gradient)
            pA = image_gradient_map(pA, gradient)
            if mode != EnumScaleMode.MATTE:
                w, h = wihi
                pA = image_scalefit(pA, w, h, mode, sample)

            if mask is not None:
                pA = image_mask_add(pA, mask)

            images.append(cv_to_tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return image_stack(images)
