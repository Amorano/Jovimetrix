""" Jovimetrix - Adjust """

from enum import Enum

import cv2

from comfy.utils import ProgressBar

from cozy_comfyui import \
    InputType, RGBAMaskType, EnumConvertType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyImageNode

from cozy_comfyui.image.adjust import \
    image_contrast, image_brightness, image_equalize, image_gamma, image_exposure, \
    image_hsv, image_invert, image_pixelate, image_pixelscale, image_posterize, \
    image_quantize, image_sharpen, image_edge_detect, image_emboss

from cozy_comfyui.image.channel import \
    channel_solid

from cozy_comfyui.image.compose import \
    image_levels, image_blend, image_mask, image_mask_add

from cozy_comfyui.image.convert import \
    tensor_to_cv, cv_to_tensor_full

from cozy_comfyui.image.misc import \
    image_stack

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

JOV_CATEGORY = "ADJUST"

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumAdjustBlur(Enum):
    BLUR = 0
    STACK_BLUR = 1
    GAUSSIAN_BLUR = 2
    MEDIAN_BLUR = 3

class EnumAdjustEdge(Enum):
    DETECT = 10
    CANNY  = 20
    LAPLACIAN = 30
    SOBEL = 40
    PREWITT = 50
    SCHARR = 60

class EnumAdjustEnhance(Enum):
    SHARPEN = 10
    EMBOSS = 20
    OUTLINE = 30

class EnumAdjustMorpho(Enum):
    DILATE = 10
    ERODE = 20
    OPEN = 30
    CLOSE = 40
    TOPHAT = 50
    BLACKHAT = 60
    GRADIENT = 70

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
    NAME = "BLUR (JOV)"
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
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 0, 0)
        params = list(zip_longest_fill(pA, op, radius))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, radius) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            # height, width = pA.shape[:2]
            if radius % 2 == 0:
                radius += 1

            match op:
                case EnumAdjustBlur.BLUR:
                    pA = cv2.blur(pA, (radius, radius))

                case EnumAdjustBlur.STACK_BLUR:
                    pA = cv2.stackBlur(pA, (radius, radius))

                case EnumAdjustBlur.GAUSSIAN_BLUR:
                    pA = cv2.GaussianBlur(pA, (radius, radius), 0)

                case EnumAdjustBlur.MEDIAN_BLUR:
                    pA = cv2.medianBlur(pA, radius)

            #pA = image_blend(pA, img_new, mask)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustEdgeNode(CozyImageNode):
    NAME = "EDGE (JOV)"
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
                    "default": EnumAdjustEdge.DETECT.name,}),
                Lexicon.RADIUS: ("INT", {
                    "default": 3, "min": 3}),
                Lexicon.ITERATION: ("INT", {
                    "default": 1, "min": 1, "max": 1000}),
                Lexicon.LOHI: ("VEC2", {
                    "default": (0, 1), "mij": 0, "maj": 1., "step": 0.01})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustEdge, EnumAdjustEdge.DETECT.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3, 3)
        count = parse_param(kw, Lexicon.ITERATION, EnumConvertType.INT, 1, 1, 1000)
        lohi = parse_param(kw, Lexicon.LOHI, EnumConvertType.VEC2, 0, 0, 1)
        params = list(zip_longest_fill(pA, op, radius, count, lohi))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, radius, count, lohi) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            # alpha = image_mask(pA)

            if radius % 2 == 0:
                radius += 1

            match op:
                case EnumAdjustEdge.DETECT:
                    lo, hi = lohi
                    pA = image_edge_detect(pA, radius, low=lo, high=hi)

                case EnumAdjustEdge.CANNY:
                    pA = image_sharpen(pA, radius, amount=count)

                case EnumAdjustEdge.LAPLACIAN:
                    pA = image_emboss(pA, count, radius)

                case EnumAdjustEdge.SOBEL:
                    pA = cv2.dilate(pA, (radius, radius), iterations=count)

                case EnumAdjustEdge.PREWITT:
                    pA = cv2.erode(pA, (radius, radius), iterations=count)

                case EnumAdjustEdge.SCHARR:
                    pA = cv2.morphologyEx(pA, cv2.MORPH_OPEN, (radius, radius), iterations=count)

            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustEnhanceNode(CozyImageNode):
    NAME = "ENHANCE (JOV)"
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
                    "default": EnumAdjustEdge.DETECT.name,}),
                Lexicon.RADIUS: ("INT", {
                    "default": 3, "min": 3}),
                Lexicon.ITERATION: ("INT", {
                    "default": 1, "min": 1, "max": 1000}),
                Lexicon.LOHI: ("VEC2", {
                    "default": (0, 1), "mij": 0, "maj": 1., "step": 0.01})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustEdge, EnumAdjustEdge.DETECT.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3, 3)
        count = parse_param(kw, Lexicon.ITERATION, EnumConvertType.INT, 1, 1, 1000)
        lohi = parse_param(kw, Lexicon.LOHI, EnumConvertType.VEC2, 0, 0, 1)
        params = list(zip_longest_fill(pA, mask, op, radius, count, lohi))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, radius, count, lohi) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            #height, width = pA.shape[:2]
            #mask = channel_solid(width, height, (255,255,255,255)) if mask is None else tensor_to_cv(mask)

            if radius % 2 == 0:
                radius += 1

            match op:
                case EnumAdjustEdge.DETECT:
                    lo, hi = lohi
                    pA = image_edge_detect(pA, low=lo, high=hi)

                case EnumAdjustEdge.SHARPEN:
                    pA = image_sharpen(pA, radius, amount=count)

                case EnumAdjustEdge.EMBOSS:
                    pA = image_emboss(pA, count, radius)

                case EnumAdjustEdge.DILATE:
                    pA = cv2.dilate(pA, (radius, radius), iterations=count)

                case EnumAdjustEdge.ERODE:
                    pA = cv2.erode(pA, (radius, radius), iterations=count)

                case EnumAdjustEdge.OPEN:
                    pA = cv2.morphologyEx(pA, cv2.MORPH_OPEN, (radius, radius), iterations=count)

                case EnumAdjustEdge.CLOSE:
                    pA = cv2.morphologyEx(pA, cv2.MORPH_CLOSE, (radius, radius), iterations=count)

                case EnumAdjustEdge.OUTLINE:
                    pA = cv2.morphologyEx(pA, cv2.MORPH_GRADIENT, (radius, radius), iterations=count)

            #pA = image_blend(pA, img_new, mask)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustLevelNode(CozyImageNode):
    NAME = "LEVELS (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """

"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.LMH: ("VEC3", {
                    "default": (0,0.5,1), "mij": 0, "maj": 1., "step": 0.01,
                    "label": ["LOW", "MID", "HIGH"]}),
                Lexicon.RANGE: ("VEC2", {
                    "default": (0, 1), "mij": 0, "maj": 1., "step": 0.01,
                    "label": ["IN", "OUT"]})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        LMH = parse_param(kw, Lexicon.LMH, EnumConvertType.VEC3, (0,0.5,1))
        inout = parse_param(kw, Lexicon.RANGE, EnumConvertType.VEC2, (0,1))
        params = list(zip_longest_fill(pA, LMH, inout))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, LMH, inout) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            #height, width = pA.shape[:2]
            #mask = channel_solid(width, height, 255) if mask is None else tensor_to_cv(mask)

            '''
            h, s, v = hsv
            img_new = image_hsv(img_new, h, s, v)
            '''
            low, mid, high = LMH
            start, end = inout
            # mid = min(high, max(mid, low))
            pA = image_levels(pA, low, mid, high, start, end)
            #print(pA.shape, img_new.shape, mask.shape)
            #pA = image_blend(pA, img_new, mask)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class AdjustLightNode(CozyImageNode):
    NAME = "LIGHT (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """

"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.BRIGHTNESS: ("FLOAT", {
                    "default": 0, "min": -1, "max": 1, "step": 0.01}),
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
        brightness = parse_param(kw, Lexicon.BRIGHTNESS, EnumConvertType.FLOAT, 0)
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

class AdjustPixelNode(CozyImageNode):
    NAME = "PIXEL (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """

"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustPixel._member_names_, {
                    "default": EnumAdjustPixel.PIXELATE.name,}),
                Lexicon.VALUE: ("INT", {
                    "default": 1, "min": 0, "max": 4096, "step": 1}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustPixel, EnumAdjustPixel.PIXELATE.name)
        val = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 0)
        params = list(zip_longest_fill(pA, op, val))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, op, val) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            #height, width = pA.shape[:2]
            #mask = channel_solid(width, height, (255,255,255,255)) if mask is None else tensor_to_cv(mask)

            match op:
                case EnumAdjustPixel.PIXELATE:
                    pA = image_pixelate(pA, val)

                case EnumAdjustPixel.PIXELSCALE:
                    pA = image_pixelscale(pA, val)

                case EnumAdjustPixel.QUANTIZE:
                    val = max(0, 256 - val)
                    pA = image_quantize(pA, val)

                case EnumAdjustPixel.POSTERIZE:
                    val = max(0, 256 - val)
                    pA = image_posterize(pA, val)

            #pA = image_blend(pA, img_new, mask)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)
