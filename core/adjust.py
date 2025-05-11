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

# EnumThreshold, EnumThresholdAdapt, image_filter, image_threshold,

from cozy_comfyui.image.adjust import \
    image_contrast, image_brightness, image_equalize, image_gamma,  \
    image_hsv, image_invert, image_pixelate, image_posterize, \
    image_quantize, image_sharpen, image_edge_detect, image_emboss

from cozy_comfyui.image.channel import \
    channel_solid

from cozy_comfyui.image.compose import \
    image_levels, image_blend

from cozy_comfyui.image.convert import \
    tensor_to_cv, cv_to_tensor_full

from cozy_comfyui.image.mask import \
    image_mask

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
    QUANTIZE = 20
    POSTERIZE = 30

# ==============================================================================
# === CLASS ===
# ==============================================================================

'''
class AdjustNode(CozyImageNode):
    NAME = "ADJUST (JOV) 🕸️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Enhance and modify images with various effects such as blurring, sharpening, color tweaks, and edge detection. Customize parameters like radius, value, and contrast, and use masks for selective effects.

Advanced options include pixelation, quantization, and morphological operations like dilation and erosion. Handle transparency effortlessly to ensure seamless blending of effects. This node is ideal for simple adjustments and complex image transformations.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.MASK: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustOP._member_names_, {
                    "default": EnumAdjustOP.BLUR.name,}),
                Lexicon.RADIUS: ("INT", {
                    "default": 3, "min": 3}),
                Lexicon.VALUE: ("FLOAT", {
                    "default": 1, "min": 0, "step": 0.01}),
                Lexicon.EDGE: ("VEC2", {
                    "default": (0, 1), "mij": 0, "maj": 1,
                    "label": ["Low", "HI"]}),
                Lexicon.LEVEL: ("VEC3", {
                    "default": (0, 0.5, 1), "mij": 0, "maj": 1,
                    "label": ["Low", "MID", "HI"],}),
                Lexicon.HSV: ("VEC3",{
                    "default": (0, 1, 1), "mij": 0, "maj": 1,
                    "label": ["H", "S", "V"],}),
                Lexicon.CONTRAST: ("FLOAT", {
                    "default": 0, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.GAMMA: ("FLOAT", {
                    "default": 1, "min": 0.00001, "max": 100, "step": 0.1}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustOP, EnumAdjustOP.BLUR.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3, 3)
        val = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 0, 0)

        edges = parse_param(kw, Lexicon.EDGE, EnumConvertType.VEC2, (0, 1), 0, 1)
        level = parse_param(kw, Lexicon.LMH, EnumConvertType.VEC3, (0, 0.5, 1), 0, 1)
        equalize = parse_param(kw, Lexicon.EQUALIZE, EnumConvertType.BOOLEAN, False)
        hsv = parse_param(kw, Lexicon.HSV, EnumConvertType.VEC3, (0, 1, 1), 0, 1)
        contrast = parse_param(kw, Lexicon.CONTRAST, EnumConvertType.FLOAT, 1, 0, 1)
        gamma = parse_param(kw, Lexicon.GAMMA, EnumConvertType.FLOAT, 1, 0, 100)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)

        params = list(zip_longest_fill(pA, mask, op, radius, val, edges, level, equalize, hsv, contrast, gamma, matte, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, radius, val, edges, level, equalize, hsv, contrast, gamma, matte, invert) in enumerate(params):
            pA = tensor_to_cv(pA) if pA is not None else channel_solid()
            img_new = image_convert(pA, 3)

            match op:
                case EnumAdjustOP.BLUR:
                    img_new = cv2.blur(img_new, (radius, radius))

                case EnumAdjustOP.STACK_BLUR:
                    r = min(radius, 1399)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.stackBlur(img_new, (r, r))

                case EnumAdjustOP.GAUSSIAN_BLUR:
                    r = min(radius, 999)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.GaussianBlur(img_new, (r, r), sigmaX=val)

                case EnumAdjustOP.MEDIAN_BLUR:
                    r = min(radius, 357)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.medianBlur(img_new, r)

                case EnumAdjustOP.SHARPEN:
                    r = min(radius, 511)
                    if r % 2 == 0:
                        r += 1
                    img_new = image_sharpen(img_new, kernel_size=r, amount=val)

                case EnumAdjustOP.EMBOSS:
                    img_new = morph_emboss(img_new, val, radius)

                case EnumAdjustOP.PIXELATE:
                    img_new = image_pixelate(img_new, val / 255.)

                case EnumAdjustOP.QUANTIZE:
                    img_new = image_quantize(img_new, int(val))

                case EnumAdjustOP.POSTERIZE:
                    img_new = image_posterize(img_new, int(val))

                case EnumAdjustOP.OUTLINE:
                    img_new = cv2.morphologyEx(img_new, cv2.MORPH_GRADIENT, (radius, radius))

                case EnumAdjustOP.DILATE:
                    img_new = cv2.dilate(img_new, (radius, radius), iterations=int(val))

                case EnumAdjustOP.ERODE:
                    img_new = cv2.erode(img_new, (radius, radius), iterations=int(val))

                case EnumAdjustOP.OPEN:
                    img_new = cv2.morphologyEx(img_new, cv2.MORPH_OPEN, (radius, radius), iterations=int(val))

                case EnumAdjustOP.CLOSE:
                    img_new = cv2.morphologyEx(img_new, cv2.MORPH_CLOSE, (radius, radius), iterations=int(val))

            h, s, v = hsv
            img_new = image_hsv(img_new, h, s, v)

            lo, hi = edges
            img_new = morph_edge_detect(img_new, low=lo, high=hi)

            if equalize:
                img_new = image_equalize(img_new)

            l, m, h = level
            img_new = image_levels(img_new, l, h, m, gamma)

            if contrast != 0:
                img_new = image_contrast(img_new, contrast)

            if gamma != 0:
                img_new = image_gamma(img_new, gamma)

            if invert:
                img_new = image_invert(img_new, val)

            if mask is not None:
                mask = tensor_to_cv(mask)

            img_new = image_blend(pA, img_new, mask)
            if pA.ndim == 3 and pA.shape[2] == 4:
                mask = image_mask(pA)
                img_new = image_convert(img_new, 4)
                img_new[:,:,3] = mask

            images.append(cv_to_tensor_full(img_new, matte))
            pbar.update_absolute(idx)
        return image_stack(images)
'''

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
                Lexicon.MASK: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustBlur._member_names_, {
                    "default": EnumAdjustBlur.BLUR.name,}),
                Lexicon.RADIUS: ("INT", {
                    "default": 3, "min": 3}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustBlur, EnumAdjustBlur.BLUR.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 0, 0)
        params = list(zip_longest_fill(pA, mask, op, radius))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, radius) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            height, width = pA.shape[:2]
            mask = channel_solid(width, height, (255,255,255,255)) if mask is None else tensor_to_cv(mask)
            if radius % 2 == 0:
                radius += 1

            match op:
                case EnumAdjustBlur.BLUR:
                    img_new = cv2.blur(pA, (radius, radius))

                case EnumAdjustBlur.STACK_BLUR:
                    img_new = cv2.stackBlur(pA, (radius, radius))

                case EnumAdjustBlur.GAUSSIAN_BLUR:
                    img_new = cv2.GaussianBlur(pA, (radius, radius))

                case EnumAdjustBlur.MEDIAN_BLUR:
                    img_new = cv2.medianBlur(pA, radius)

            pA = image_blend(pA, img_new, mask)
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
                Lexicon.MASK: (COZY_TYPE_IMAGE, {}),
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
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustEdge, EnumAdjustEdge.DETECT.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3, 3)
        count = parse_param(kw, Lexicon.ITERATION, EnumConvertType.INT, 1, 1, 1000)
        lohi = parse_param(kw, Lexicon.LOHI, EnumConvertType.VEC2, 0, 0, 1)
        params = list(zip_longest_fill(pA, mask, op, radius, count, lohi))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, radius, count, lohi) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            alpha = image_mask(pA)
            height, width = pA.shape[:2]
            mask = channel_solid(width, height, (255,255,255,255)) if mask is None else tensor_to_cv(mask, chan=1)

            if radius % 2 == 0:
                radius += 1

            match op:
                case EnumAdjustEdge.DETECT:
                    lo, hi = lohi
                    img_new = image_edge_detect(pA, radius, low=lo, high=hi)

                case EnumAdjustEdge.CANNY:
                    img_new = image_sharpen(pA, radius, amount=count)

                case EnumAdjustEdge.LAPLACIAN:
                    img_new = image_emboss(pA, count, radius)

                case EnumAdjustEdge.SOBEL:
                    img_new = cv2.dilate(pA, (radius, radius), iterations=count)

                case EnumAdjustEdge.PREWITT:
                    img_new = cv2.erode(pA, (radius, radius), iterations=count)

                case EnumAdjustEdge.SCHARR:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_OPEN, (radius, radius), iterations=count)

            pA = image_blend(pA, img_new, mask)
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
                Lexicon.MASK: (COZY_TYPE_IMAGE, {}),
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
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustEdge, EnumAdjustEdge.DETECT.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3, 3)
        count = parse_param(kw, Lexicon.ITERATION, EnumConvertType.INT, 1, 1, 1000)
        lohi = parse_param(kw, Lexicon.LOHI, EnumConvertType.VEC2, 0, 0, 1)
        params = list(zip_longest_fill(pA, mask, op, radius, count, lohi))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, radius, count, lohi) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            height, width = pA.shape[:2]
            mask = channel_solid(width, height, (255,255,255,255)) if mask is None else tensor_to_cv(mask)

            if radius % 2 == 0:
                radius += 1

            match op:
                case EnumAdjustEdge.DETECT:
                    lo, hi = lohi
                    img_new = image_edge_detect(pA, low=lo, high=hi)

                case EnumAdjustEdge.SHARPEN:
                    img_new = image_sharpen(pA, radius, amount=count)

                case EnumAdjustEdge.EMBOSS:
                    img_new = image_emboss(pA, count, radius)

                case EnumAdjustEdge.DILATE:
                    img_new = cv2.dilate(pA, (radius, radius), iterations=count)

                case EnumAdjustEdge.ERODE:
                    img_new = cv2.erode(pA, (radius, radius), iterations=count)

                case EnumAdjustEdge.OPEN:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_OPEN, (radius, radius), iterations=count)

                case EnumAdjustEdge.CLOSE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_CLOSE, (radius, radius), iterations=count)

                case EnumAdjustEdge.OUTLINE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_GRADIENT, (radius, radius), iterations=count)

            pA = image_blend(pA, img_new, mask)
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
                Lexicon.MASK: (COZY_TYPE_IMAGE, {}),
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
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        LMH = parse_param(kw, Lexicon.LMH, EnumConvertType.VEC3, (0,0.5,1))
        inout = parse_param(kw, Lexicon.RANGE, EnumConvertType.VEC2, (0,1))

        params = list(zip_longest_fill(pA, mask, LMH, inout))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, LMH, inout) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            height, width = pA.shape[:2]
            mask = channel_solid(width, height, 255) if mask is None else tensor_to_cv(mask)

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
                Lexicon.MASK: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustLight._member_names_, {
                    "default": EnumAdjustLight.CONTRAST.name,}),
                Lexicon.VALUE: ("FLOAT", {
                    "default": 0, "min": -1, "max": 1, "step": 0.001}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustLight, EnumAdjustLight.CONTRAST.name)
        val = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 0)

        params = list(zip_longest_fill(pA, mask, op, val))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, val) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            height, width = pA.shape[:2]
            mask = channel_solid(width, height, (255,255,255,255)) if mask is None else tensor_to_cv(mask)

            match op:
                case EnumAdjustLight.BRIGHTNESS:
                    img_new = image_contrast(pA, val)

                case EnumAdjustLight.CONTRAST:
                    img_new = image_contrast(pA, val)

                case EnumAdjustLight.EQUALIZE:
                    img_new = image_equalize(pA)

                case EnumAdjustLight.EXPOSURE:
                    img_new = image_contrast(pA, val)

                case EnumAdjustLight.GAMMA:
                    img_new = image_gamma(pA, val)

            '''
            h, s, v = hsv
            img_new = image_hsv(img_new, h, s, v)

            l, m, h = level
            img_new = image_levels(img_new, l, h, m, gamma)
            '''

            pA = image_blend(pA, img_new, mask)
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
                Lexicon.MASK: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumAdjustPixel._member_names_, {
                    "default": EnumAdjustPixel.PIXELATE.name,}),
                Lexicon.VALUE: ("INT", {
                    "default": 1, "min": 0, "max": 255, "step": 1}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        op = parse_param(kw, Lexicon.FUNCTION, EnumAdjustPixel, EnumAdjustPixel.PIXELATE.name)
        val = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 0, 255)
        params = list(zip_longest_fill(pA, mask, op, val))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, val) in enumerate(params):
            pA = channel_solid() if pA is None else tensor_to_cv(pA)
            height, width = pA.shape[:2]
            mask = channel_solid(width, height, (255,255,255,255)) if mask is None else tensor_to_cv(mask)

            match op:
                case EnumAdjustPixel.PIXELATE:
                    img_new = image_pixelate(pA, val / 255.)

                case EnumAdjustPixel.QUANTIZE:
                    img_new = image_quantize(pA, val)

                case EnumAdjustPixel.POSTERIZE:
                    img_new = image_posterize(pA, val)

            pA = image_blend(pA, img_new, mask)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)
