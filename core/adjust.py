""" Jovimetrix - Adjust """

import sys
from enum import Enum
from typing import Any
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, io
from comfy.utils import ProgressBar

from cozy_comfyui import \
    InputType, RGBAMaskType, EnumConvertType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfy.node import \
    COZY_TYPE_IMAGE as COZY_TYPE_IMAGEv3, \
    CozyImageNode as CozyImageNodev3

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyImageNode

from cozy_comfyui.image.adjust import \
    EnumAdjustBlur, EnumAdjustColor, EnumAdjustEdge, EnumAdjustMorpho, \
    image_contrast, image_brightness, image_equalize, image_gamma, \
    image_exposure, image_pixelate, image_pixelscale, \
    image_posterize, image_quantize, image_sharpen, image_morphology, \
    image_emboss, image_blur, image_edge, image_color, \
    image_autolevel, image_autolevel_histogram

from cozy_comfyui.image.channel import \
    channel_solid

from cozy_comfyui.image.compose import \
    image_levels

from cozy_comfyui.image.convert import \
    tensor_to_cv, cv_to_tensor_full, image_mask, image_mask_add

from cozy_comfyui.image.misc import \
    image_stack

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

JOV_CATEGORY = "ADJUST"

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumAutoLevel(Enum):
    MANUAL = 10
    AUTO = 20
    HISTOGRAM = 30

class EnumAdjustLight(Enum):
    EXPOSURE = 10
    GAMMA = 20
    BRIGHTNESS = 30
    CONTRAST = 40
    EQUALIZE = 50

class EnumAdjustPixel(Enum):
    PIXELATE = 10
    PIXELSCALE = 20
    QUANTIZE = 30
    POSTERIZE = 40

# ==============================================================================
# === CLASS ===
# ==============================================================================

class AdjustBlurNode(CozyImageNode):
    NAME = "ADJUST: BLUR (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Enhance and modify images with various blur effects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustBlur._member_names_, {
                    "default": EnumAdjustBlur.BLUR.name,}),
                Lexicon.RADIUS: ("INT", {
                    "default": 3, "min": 3}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustBlur, EnumAdjustBlur.BLUR.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3)
        params = list(zip_longest_fill(pA, op, radius))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, radius) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            # height, width = pA.shape[:2]
            pA = image_blur(pA, op, radius)
            #pA = image_blend(pA, img_new, mask)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustColorNode(CozyImageNode):
    NAME = "ADJUST: COLOR (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Enhance and modify images with various blur effects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustColor._member_names_, {
                    "default": EnumAdjustColor.RGB.name,}),
                Lexicon.VEC: ("VEC3", {
                    "default": (0,0,0), "mij": -1, "maj": 1, "step": 0.025})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustColor, EnumAdjustColor.RGB.name)
        vec = parse_param(kw, Lexicon.VEC, EnumConvertType.VEC3, (0,0,0))
        params = list(zip_longest_fill(pA, op, vec))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, vec) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            pA = image_color(pA, op, vec[0], vec[1], vec[2])
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustEdgeNode(CozyImageNode):
    NAME = "ADJUST: EDGE (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Enhanced edge detection.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustEdge._member_names_, {
                    "default": EnumAdjustEdge.CANNY.name,}),
                Lexicon.RADIUS: ("INT", {
                    "default": 1, "min": 1}),
                Lexicon.ITERATION: ("INT", {
                    "default": 1, "min": 1, "max": 1000}),
                Lexicon.LOHI: ("VEC2", {
                    "default": (0, 1), "mij": 0, "maj": 1, "step": 0.01})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustEdge, EnumAdjustEdge.CANNY.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 1)
        count = parse_param(kw, Lexicon.ITERATION, EnumConvertType.INT, 1)
        lohi = parse_param(kw, Lexicon.LOHI, EnumConvertType.VEC2, (0,1))
        params = list(zip_longest_fill(pA, op, radius, count, lohi))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, radius, count, lohi) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            alpha = image_mask(pA)
            pA = image_edge(pA, op, radius, count, lohi[0], lohi[1])
            pA = image_mask_add(pA, alpha)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustEmbossNode(CozyImageNode):
    NAME = "ADJUST: EMBOSS (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Emboss boss mode.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.HEADING: ("FLOAT", {
                    "default": -45, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.1}),
                Lexicon.ELEVATION: ("FLOAT", {
                    "default": 45, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.1}),
                Lexicon.DEPTH: ("FLOAT", {
                    "default": 10, "min": 0, "max": sys.float_info.max, "step": 0.1,
                    "tooltip": "Depth perceived from the light angles above"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        heading = parse_param(kw, Lexicon.HEADING, EnumConvertType.FLOAT, -45)
        elevation = parse_param(kw, Lexicon.ELEVATION, EnumConvertType.FLOAT, 45)
        depth = parse_param(kw, Lexicon.DEPTH, EnumConvertType.FLOAT, 10)
        params = list(zip_longest_fill(pA, heading, elevation, depth))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, heading, elevation, depth) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            alpha = image_mask(pA)
            pA = image_emboss(pA, heading, elevation, depth)
            pA = image_mask_add(pA, alpha)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustLevelNode(CozyImageNode):
    NAME = "ADJUST: LEVELS (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Manual or automatic adjust image levels so that the darkest pixel becomes black
and the brightest pixel becomes white, enhancing overall contrast.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.LMH: ("VEC3", {
                    "default": (0,0.5,1), "mij": 0, "maj": 1, "step": 0.01,
                    "label": ["LOW", "MID", "HIGH"]}),
                Lexicon.RANGE: ("VEC2", {
                    "default": (0, 1), "mij": 0, "maj": 1, "step": 0.01,
                    "label": ["IN", "OUT"]}),
                Lexicon.MODE: (EnumAutoLevel._member_names_, {
                    "default": EnumAutoLevel.MANUAL.name,
                    "tooltip": "Autolevel linearly or with Histogram bin values, per channel"
                }),
                "clip": ("FLOAT", {
                    "default": 0.5, "min": 0, "max": 1.0, "step": 0.01
                })
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        LMH = parse_param(kw, Lexicon.LMH, EnumConvertType.VEC3, (0,0.5,1))
        inout = parse_param(kw, Lexicon.RANGE, EnumConvertType.VEC2, (0,1))
        mode = parse_param(kw, Lexicon.MODE, EnumAutoLevel, EnumAutoLevel.AUTO.name)
        clip = parse_param(kw, "clip", EnumConvertType.FLOAT, 0.5, 0, 1)
        params = list(zip_longest_fill(pA, LMH, inout, mode, clip))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, LMH, inout, mode, clip) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            '''
            h, s, v = hsv
            img_new = image_hsv(img_new, h, s, v)
            '''
            match mode:
                case EnumAutoLevel.MANUAL:
                    low, mid, high = LMH
                    start, end = inout
                    pA = image_levels(pA, low, mid, high, start, end)

                case EnumAutoLevel.AUTO:
                    pA = image_autolevel(pA)

                case EnumAutoLevel.HISTOGRAM:
                    pA = image_autolevel_histogram(pA, clip)

            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustLightNode(CozyImageNode):
    NAME = "ADJUST: LIGHT (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Tonal adjustments. They can be applied individually or all at the same time in order: brightness, contrast, histogram equalization, exposure, and gamma correction.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.BRIGHTNESS: ("FLOAT", {
                    "default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.CONTRAST: ("FLOAT", {
                    "default": 0, "min": -1, "max": 1, "step": 0.01}),
                Lexicon.EQUALIZE: ("BOOLEAN", {
                    "default": False}),
                Lexicon.EXPOSURE: ("FLOAT", {
                    "default": 1, "min": -8, "max": 8, "step": 0.01}),
                Lexicon.GAMMA: ("FLOAT", {
                    "default": 1, "min": 0, "max": 8, "step": 0.01}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        brightness = parse_param(kw, Lexicon.BRIGHTNESS, EnumConvertType.FLOAT, 0.5)
        contrast = parse_param(kw, Lexicon.CONTRAST, EnumConvertType.FLOAT, 0)
        equalize = parse_param(kw, Lexicon.EQUALIZE, EnumConvertType.FLOAT, 0)
        exposure = parse_param(kw, Lexicon.EXPOSURE, EnumConvertType.FLOAT, 0)
        gamma = parse_param(kw, Lexicon.GAMMA, EnumConvertType.FLOAT, 0)
        params = list(zip_longest_fill(pA, brightness, contrast, equalize, exposure, gamma))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, brightness, contrast, equalize, exposure, gamma) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            alpha = image_mask(pA)

            brightness = 2. * (brightness - 0.5)
            if brightness != 0:
                pA = image_brightness(pA, brightness)

            if contrast != 0:
                pA = image_contrast(pA, contrast)

            if equalize:
                pA = image_equalize(pA)

            if exposure != 1:
                pA = image_exposure(pA, exposure)

            if gamma != 1:
                pA = image_gamma(pA, gamma)

            '''
            h, s, v = hsv
            img_new = image_hsv(img_new, h, s, v)

            l, m, h = level
            img_new = image_levels(img_new, l, h, m, gamma)
            '''
            pA = image_mask_add(pA, alpha)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustMorphNode(CozyImageNode):
    NAME = "ADJUST: MORPHOLOGY (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Operations based on the image shape.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustMorpho._member_names_, {
                    "default": EnumAdjustMorpho.DILATE.name,}),
                Lexicon.RADIUS: ("INT", {
                    "default": 1, "min": 1}),
                Lexicon.ITERATION: ("INT", {
                    "default": 1, "min": 1, "max": 1000}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustMorpho, EnumAdjustMorpho.DILATE.name)
        kernel = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 1)
        count = parse_param(kw, Lexicon.ITERATION, EnumConvertType.INT, 1)
        params = list(zip_longest_fill(pA, op, kernel, count))
        images: list[Any] = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, kernel, count) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            alpha = image_mask(pA)
            pA = image_morphology(pA, op, kernel, count)
            pA = image_mask_add(pA, alpha)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustPixelNode(CozyImageNode):
    NAME = "ADJUST: PIXEL (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Pixel-level transformations. The val parameter controls the intensity or resolution of the effect, depending on the operation.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustPixel._member_names_, {
                    "default": EnumAdjustPixel.PIXELATE.name,}),
                Lexicon.VALUE: ("FLOAT", {
                    "default": 0, "min": 0, "max": 1, "step": 0.01})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustPixel, EnumAdjustPixel.PIXELATE.name)
        val = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 0)
        params = list(zip_longest_fill(pA, op, val))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, val) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA, chan=4)
            alpha = image_mask(pA)

            match op:
                case EnumAdjustPixel.PIXELATE:
                    pA = image_pixelate(pA, val / 2.)

                case EnumAdjustPixel.PIXELSCALE:
                    pA = image_pixelscale(pA, val)

                case EnumAdjustPixel.QUANTIZE:
                    pA = image_quantize(pA, val)

                case EnumAdjustPixel.POSTERIZE:
                    pA = image_posterize(pA, val)

            pA = image_mask_add(pA, alpha)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustSharpenNode(CozyImageNode):
    NAME = "ADJUST: SHARPEN (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Sharpen the pixels of an image.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.AMOUNT: ("FLOAT", {
                    "default": 0, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.THRESHOLD: ("FLOAT", {
                    "default": 0, "min": 0, "max": 1, "step": 0.01})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        amount = parse_param(kw, Lexicon.AMOUNT, EnumConvertType.FLOAT, 0)
        threshold = parse_param(kw, Lexicon.THRESHOLD, EnumConvertType.FLOAT, 0)
        params = list(zip_longest_fill(pA, amount, threshold))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, amount, threshold) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            pA = image_sharpen(pA, amount / 2., threshold=threshold / 25.5)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustSharpenNodev3(CozyImageNodev3):
    @classmethod
    def define_schema(cls, **kwarg) -> io.Schema:
        schema = super(**kwarg).define_schema()
        schema.display_name = "ADJUST: SHARPEN (JOV)"
        schema.category = JOV_CATEGORY
        schema.description = "Sharpen the pixels of an image."

        schema.inputs.extend([
            io.MultiType.Input(
                id=Lexicon.IMAGE[0],
                types=COZY_TYPE_IMAGEv3,
                display_name=Lexicon.IMAGE[0],
                optional=True,
                tooltip=Lexicon.IMAGE[1]
            ),
            io.Float.Input(
                id=Lexicon.AMOUNT[0],
                display_name=Lexicon.AMOUNT[0],
                optional=True,
                default= 0,
                min=0,
                max=1,
                step=0.01,
                tooltip=Lexicon.AMOUNT[1]
            ),
            io.Float.Input(
                id=Lexicon.THRESHOLD[0],
                display_name=Lexicon.THRESHOLD[0],
                optional=True,
                default= 0,
                min=0,
                max=1,
                step=0.01,
                tooltip=Lexicon.THRESHOLD[1]
            )

        ])
        return schema

    @classmethod
    def execute(self, *arg, **kw) -> io.NodeOutput:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        amount = parse_param(kw, Lexicon.AMOUNT, EnumConvertType.FLOAT, 0)
        threshold = parse_param(kw, Lexicon.THRESHOLD, EnumConvertType.FLOAT, 0)
        params = list(zip_longest_fill(pA, amount, threshold))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, amount, threshold) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            pA = image_sharpen(pA, amount / 2., threshold=threshold / 25.5)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return io.NodeOutput(image_stack(images))

class AdjustExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AdjustSharpenNodev3
        ]

async def comfy_entrypoint() -> AdjustExtension:
    return AdjustExtension()