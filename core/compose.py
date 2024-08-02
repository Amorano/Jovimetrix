"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition
"""

from enum import Enum
from typing import Any, List, Tuple

import cv2
import torch
import numpy as np

from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOV_TYPE_IMAGE, JOVBaseNode, JOVImageNode, Lexicon
from Jovimetrix.sup.util import parse_dynamic, parse_param, \
    zip_longest_fill, EnumConvertType
from Jovimetrix.sup.image import  \
    channel_merge, channel_solid, channel_swap, color_match_histogram, \
    color_match_lut, image_filter, image_gradient_map, image_minmax,  \
    image_quantize, image_scalefit, \
    color_match_reinhard, cv2tensor_full, image_color_blind, image_contrast,\
    image_crop, image_crop_center, image_crop_polygonal, image_equalize, \
    image_gamma, image_grayscale, image_hsv, image_levels, image_convert, \
    image_mask, image_mask_add, image_matte, image_pixelate, image_posterize, \
    image_sharpen, image_threshold, image_transform, image_edge_wrap, \
    image_split, morph_edge_detect, morph_emboss, pixel_eval, tensor2cv, \
    color_theory, remap_fisheye, remap_perspective, remap_polar, cv2tensor, \
    remap_sphere, image_invert, image_stack, image_mirror, image_blend, \
    EnumImageType, EnumColorTheory, EnumProjection, EnumScaleMode, \
    EnumEdge, EnumMirrorMode, EnumOrientation, EnumPixelSwizzle, EnumBlendType, \
    EnumCBDeficiency, EnumCBSimulator, EnumColorMap, EnumAdjustOP, \
    EnumThreshold, EnumInterpolation, EnumThresholdAdapt, \
    MIN_IMAGE_SIZE

# =============================================================================

JOV_CATEGORY = "COMPOSE"

class EnumColorMatchMode(Enum):
    REINHARD = 30
    LUT = 10
    HISTOGRAM = 20

class EnumColorMatchMap(Enum):
    USER_MAP = 0
    PRESET_MAP = 10

class EnumCropMode(Enum):
    CENTER = 20
    XY = 0
    FREE = 10
    HEAD = 15
    BODY = 25

# =============================================================================

class AdjustNode(JOVImageNode):
    NAME = "ADJUST (JOV) 🕸️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Enhance and modify images with various effects such as blurring, sharpening, color tweaks, and edge detection. Customize parameters like radius, value, and contrast, and use masks for selective effects. Advanced options include pixelation, quantization, and morphological operations like dilation and erosion. Handle transparency effortlessly to ensure seamless blending of effects. This node is ideal for simple adjustments and complex image transformations.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.MASK: (JOV_TYPE_IMAGE, {}),
                Lexicon.FUNC: (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name,
                                                            "tooltip":"Type of adjustment (e.g., blur, sharpen, invert)"}),
                Lexicon.RADIUS: ("INT", {"default": 3, "min": 3}),
                Lexicon.VALUE: ("FLOAT", {"default": 1, "min": 0}),
                Lexicon.LOHI: ("VEC2", {"default": (0, 1),
                                        "min": 0, "max": 1, "label": [Lexicon.LO, Lexicon.HI]}),
                Lexicon.LMH: ("VEC3", {"default": (0, 0.5, 1),
                                        "min": 0, "max": 1, "label": [Lexicon.LO, Lexicon.MID, Lexicon.HI]}),
                Lexicon.HSV: ("VEC3",{"default": (0, 1, 1),
                                    "min": 0, "max": 1,  "label": [Lexicon.H, Lexicon.S, Lexicon.V]}),
                Lexicon.CONTRAST: ("FLOAT", {"default": 0, "min": 0, "max": 1}),
                Lexicon.GAMMA: ("FLOAT", {"default": 1, "min": 0.00001, "max": 1}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, ...]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumAdjustOP.BLUR.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3, 3)
        val = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 0, 0)
        lohi = parse_param(kw, Lexicon.LOHI, EnumConvertType.VEC2, [(0, 1)], 0, 1)
        lmh = parse_param(kw, Lexicon.LMH, EnumConvertType.VEC3, [(0, 0.5, 1)], 0, 1)
        hsv = parse_param(kw, Lexicon.HSV, EnumConvertType.VEC3, [(0, 1, 1)], 0, 1)
        contrast = parse_param(kw, Lexicon.CONTRAST, EnumConvertType.FLOAT, 1, 0, 1)
        gamma = parse_param(kw, Lexicon.GAMMA, EnumConvertType.FLOAT, 1, 0, 1)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, mask, op, radius, val, lohi,
                                                    lmh, hsv, contrast, gamma, matte, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, radius, val, lohi, lmh, hsv, contrast, gamma, matte, invert) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            cc = pA.shape[2] if pA.ndim == 3 else 1
            if cc == 4:
                alpha = pA[..., 3]

            match EnumAdjustOP[op]:
                case EnumAdjustOP.INVERT:
                    img_new = image_invert(pA, val)

                case EnumAdjustOP.LEVELS:
                    l, m, h = lmh
                    img_new = image_levels(pA, l, h, m, gamma)

                case EnumAdjustOP.HSV:
                    h, s, v = hsv
                    img_new = image_hsv(pA, h, s, v)
                    if contrast != 0:
                        img_new = image_contrast(img_new, 1 - contrast)

                    if gamma != 0:
                        img_new = image_gamma(img_new, gamma)

                case EnumAdjustOP.FIND_EDGES:
                    lo, hi = lohi
                    img_new = morph_edge_detect(pA, low=lo, high=hi)

                case EnumAdjustOP.BLUR:
                    img_new = cv2.blur(pA, (radius, radius))

                case EnumAdjustOP.STACK_BLUR:
                    r = min(radius, 1399)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.stackBlur(pA, (r, r))

                case EnumAdjustOP.GAUSSIAN_BLUR:
                    r = min(radius, 999)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.GaussianBlur(pA, (r, r), sigmaX=val)

                case EnumAdjustOP.MEDIAN_BLUR:
                    r = min(radius, 357)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.medianBlur(pA, r)

                case EnumAdjustOP.SHARPEN:
                    r = min(radius, 511)
                    if r % 2 == 0:
                        r += 1
                    img_new = image_sharpen(pA, kernel_size=r, amount=val)

                case EnumAdjustOP.EMBOSS:
                    img_new = morph_emboss(pA, val, radius)

                case EnumAdjustOP.EQUALIZE:
                    img_new = image_equalize(pA)

                case EnumAdjustOP.PIXELATE:
                    img_new = image_pixelate(pA, val / 255.)

                case EnumAdjustOP.QUANTIZE:
                    img_new = image_quantize(pA, int(val))

                case EnumAdjustOP.POSTERIZE:
                    img_new = image_posterize(pA, int(val))

                case EnumAdjustOP.OUTLINE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_GRADIENT, (radius, radius))

                case EnumAdjustOP.DILATE:
                    img_new = cv2.dilate(pA, (radius, radius), iterations=int(val))

                case EnumAdjustOP.ERODE:
                    img_new = cv2.erode(pA, (radius, radius), iterations=int(val))

                case EnumAdjustOP.OPEN:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_OPEN, (radius, radius), iterations=int(val))

                case EnumAdjustOP.CLOSE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_CLOSE, (radius, radius), iterations=int(val))

            h, w = pA.shape[:2]
            mask = channel_solid(w, h, 255) if mask is None else tensor2cv(mask)
            mask = image_grayscale(mask)
            if invert:
                mask = 255 - mask
            pA = image_blend(pA, img_new, mask)
            if cc == 4:
                pA[..., 3] = alpha
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class BlendNode(JOVImageNode):
    NAME = "BLEND (JOV) ⚗️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 10
    DESCRIPTION = """
Combine two input images using various blending modes, such as normal, screen, multiply, overlay, etc. It also supports alpha blending and masking to achieve complex compositing effects. This node is essential for creating layered compositions and adding visual richness to images.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {"tooltip": "Background Plate"}),
                Lexicon.PIXEL_B: (JOV_TYPE_IMAGE, {"tooltip": "Image to Overlay on Background Plate"}),
                Lexicon.MASK: (JOV_TYPE_IMAGE, {"tooltip": "Optional Mask to use for Alpha Blend Operation. If empty, will use the ALPHA of B"}),
                Lexicon.FUNC: (EnumBlendType._member_names_, {"default": EnumBlendType.NORMAL.name, "tooltip": "Blending Operation"}),
                Lexicon.A: ("FLOAT", {"default": 1, "min": 0, "max": 1, "tooltip": "Amount of Blending to Perform on the Selected Operation"}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "min":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        func = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumBlendType.NORMAL.name)
        alpha = parse_param(kw, Lexicon.A, EnumConvertType.FLOAT, 1, 0, 1)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, EnumScaleMode.NONE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumConvertType.STRING, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, pB, mask, func, alpha, flip, mode, wihi, sample, matte, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, mask, func, alpha, flip, mode, wihi, sample, matte, invert) in enumerate(params):
            if flip:
                pA, pB = pB, pA

            w, h = MIN_IMAGE_SIZE, MIN_IMAGE_SIZE
            if pA is not None:
                h, w = pA.shape[:2]
            elif pB is not None:
                h, w = pB.shape[:2]
            elif mask is not None:
                h, w = mask.shape[:2]

            tmask = None
            if pA is None:
                pA = channel_solid(w, h, matte, chan=EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)
                matted = pixel_eval(matte, EnumImageType.BGRA)
                pA = image_matte(pA, matted)
                tmask = pA

            if pB is None:
                pB = channel_solid(w, h, matte, chan=EnumImageType.BGRA)
            else:
                pB = tensor2cv(pB)
                tmask = pB

            if mask is None:
                mask = channel_solid(w, h, matte[3], EnumImageType.GRAYSCALE) if tmask is None else image_mask(tmask)
            else:
                mask = tensor2cv(mask)
                mask = image_grayscale(mask)

            if invert:
                mask = 255 - mask

            func = EnumBlendType[func]
            img = image_blend(pA, pB, mask, func, alpha)
            mode = EnumScaleMode[mode]
            if mode != EnumScaleMode.NONE:
                w, h = wihi
                sample = EnumInterpolation[sample]
                img = image_scalefit(img, w, h, mode, sample)
            img = cv2tensor_full(img, matte)
            images.append(img)
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class ColorBlindNode(JOVImageNode):
    NAME = "COLOR BLIND (JOV) 👁‍🗨"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Simulate color blindness effects on images. You can select various types of color deficiencies, adjust the severity of the effect, and apply the simulation using different simulators. This node is ideal for accessibility testing and design adjustments, ensuring inclusivity in your visual content.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.DEFICIENCY: (EnumCBDeficiency._member_names_,
                                            {"default": EnumCBDeficiency.PROTAN.name}),
                Lexicon.SIMULATOR: (EnumCBSimulator._member_names_,
                                            {"default": EnumCBSimulator.AUTOSELECT.name}),
                Lexicon.VALUE: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.001, "tooltip":"alpha blending"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        deficiency = parse_param(kw, Lexicon.DEFICIENCY, EnumConvertType.STRING, EnumCBDeficiency.PROTAN.name)
        simulator = parse_param(kw, Lexicon.SIMULATOR, EnumConvertType.STRING, EnumCBSimulator.AUTOSELECT.name)
        severity = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 1)
        params = list(zip_longest_fill(pA, deficiency, simulator, severity))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, deficiency, simulator, severity) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor2cv(pA)
            deficiency = EnumCBDeficiency[deficiency]
            simulator = EnumCBSimulator[simulator]
            pA = image_color_blind(pA, deficiency, simulator, severity)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class ColorMatchNode(JOVImageNode):
    NAME = "COLOR MATCH (JOV) 💞"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Adjust the color scheme of one image to match another with the Color Match Node. Choose from various color matching modes, including LUT, Histogram, and Reinhard. You can specify options like color maps, the number of colors, and whether to flip or invert the images. This node allows for the creation of seamless and cohesive visuals, making it ideal for texture work or masking in motion graphics and design projects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {}),
                Lexicon.PIXEL_B: (JOV_TYPE_IMAGE, {}),
                Lexicon.COLORMATCH_MODE: (EnumColorMatchMode._member_names_,
                                            {"default": EnumColorMatchMode.REINHARD.name}),
                Lexicon.COLORMATCH_MAP: (EnumColorMatchMap._member_names_,
                                            {"default": EnumColorMatchMap.USER_MAP.name}),
                Lexicon.COLORMAP: (EnumColorMap._member_names_,
                                    {"default": EnumColorMap.HSV.name}),
                Lexicon.VALUE: ("INT", {"default": 255, "min": 0, "max": 255}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False,
                                                "tooltip": "Invert the color match output"}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        colormatch_mode = parse_param(kw, Lexicon.COLORMATCH_MODE, EnumConvertType.STRING, EnumColorMatchMode.REINHARD.name)
        colormatch_map = parse_param(kw, Lexicon.COLORMATCH_MAP, EnumConvertType.STRING, EnumColorMatchMap.USER_MAP.name)
        colormap = parse_param(kw, Lexicon.COLORMAP, EnumConvertType.STRING, EnumColorMap.HSV.name)
        num_colors = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 255)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, pB, colormap, colormatch_mode, colormatch_map, num_colors, flip, invert, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, colormap, mode, cmap, num_colors, flip, invert, matte) in enumerate(params):
            if flip == True:
                pA, pB = pB, pA
            cc = 3
            if pA is None:
                pA = channel_solid(chan=EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)
                if (cc := pA.shape[2] if len(pA.shape) > 2 else 1) == 4:
                    mask = image_mask(pA)
                pA = image_convert(pA, 4)
            h, w = pA.shape[:2]
            if pB is None:
                pB = channel_solid(chan=EnumImageType.BGRA)
            else:
                pB = tensor2cv(pB)
                pB = image_convert(pB, 4)
            mode = EnumColorMatchMode[mode]
            match mode:
                case EnumColorMatchMode.LUT:
                    cmap = EnumColorMatchMap[cmap]
                    if cmap == EnumColorMatchMap.PRESET_MAP:
                        pB = None
                    colormap = EnumColorMap[colormap]
                    pA = color_match_lut(pA, colormap.value, pB, num_colors)
                case EnumColorMatchMode.HISTOGRAM:
                    pB = image_scalefit(pB, w, h, EnumScaleMode.CROP)
                    pB = image_scalefit(pB, w, h, EnumScaleMode.MATTE)
                    pA = color_match_histogram(pA, pB)
                case EnumColorMatchMode.REINHARD:
                    pA = color_match_reinhard(pA, pB)

            if invert == True:
                pA = image_invert(pA, 1)
            if cc == 4:
                pA = image_mask_add(pA, mask)

            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class ColorTheoryNode(JOVBaseNode):
    NAME = "COLOR THEORY (JOV) 🛞"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (Lexicon.C1, Lexicon.C2, Lexicon.C3, Lexicon.C4, Lexicon.C5)
    SORT = 100
    DESCRIPTION = """
Generate a color harmony based on the selected scheme. Supported schemes include complimentary, analogous, triadic, tetradic, and more. Users can customize the angle of separation for color calculations, offering flexibility in color manipulation and exploration of different color palettes.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.SCHEME: (EnumColorTheory._member_names_, {"default": EnumColorTheory.COMPLIMENTARY.name}),
                Lexicon.VALUE: ("INT", {"default": 45, "min": -90, "max": 90, "tooltip": "Custom angle of separation to use when calculating colors"}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        scheme = parse_param(kw, Lexicon.SCHEME, EnumConvertType.STRING, EnumColorTheory.COMPLIMENTARY.name)
        user = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 0, -180, 180)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, scheme, user, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (img, target, user, invert) in enumerate(params):
            img = tensor2cv(img) if img is not None else channel_solid(chan=EnumImageType.BGRA)
            target = EnumColorTheory[target]
            img = color_theory(img, user, target)
            if invert:
                img = (image_invert(s, 1) for s in img)
            images.append([cv2tensor(a) for a in img])
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class CropNode(JOVImageNode):
    NAME = "CROP (JOV) ✂️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 5
    DESCRIPTION = """
Extract a portion of an input image or resize it. It supports various cropping modes, including center cropping, custom XY cropping, and free-form polygonal cropping. This node is useful for preparing image data for specific tasks or extracting regions of interest.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.FUNC: (EnumCropMode._member_names_, {"default": EnumCropMode.CENTER.name}),
                Lexicon.XY: ("VEC2", {"default": (0, 0), "min": 0.5, "max": 0.5, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "min": MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.TLTR: ("VEC4", {"default": (0, 0, 0, 1), "min": 0, "max": 1, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
                Lexicon.BLBR: ("VEC4", {"default": (1, 0, 1, 1), "min": 0, "max": 1,  "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        func = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumCropMode.CENTER.name)
        # if less than 1 then use as scalar, over 1 = int(size)
        xy = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, [(0, 0,)], 1)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        tltr = parse_param(kw, Lexicon.TLTR, EnumConvertType.VEC4, [(0, 0, 0, 1,)], 0, 1)
        blbr = parse_param(kw, Lexicon.BLBR, EnumConvertType.VEC4, [(1, 0, 1, 1,)], 0, 1)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, func, xy, wihi, tltr, blbr, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, func, xy, wihi, tltr, blbr, matte) in enumerate(params):
            width, height = wihi
            pA = tensor2cv(pA) if pA is not None else channel_solid(width, height)
            alpha = None
            if pA.ndim == 3 and pA.shape[2] == 4:
                alpha = image_mask(pA)

            func = EnumCropMode[func]
            if func == EnumCropMode.FREE:
                x1, y1, x2, y2 = tltr
                x4, y4, x3, y3 = blbr
                points = (x1 * width, y1 * height), (x2 * width, y2 * height), \
                    (x3 * width, y3 * height), (x4 * width, y4 * height)
                pA = image_crop_polygonal(pA, points)
                if alpha is not None:
                    alpha = image_crop_polygonal(alpha, points)
                    pA[..., 3] = alpha[..., 0][:,:]

            elif func == EnumCropMode.XY:
                pA = image_crop(pA, width, height, xy)
            elif func == EnumCropMode.HEAD:
                pass
            elif func == EnumCropMode.BODY:
                pass
            else:
                pA = image_crop_center(pA, width, height)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class FilterMaskNode(JOVImageNode):
    NAME = "FILTER MASK (JOV) 🤿"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 700
    DESCRIPTION = """
Create masks based on specific color ranges within an image. Specify the color range using start and end values and an optional fuzziness factor to adjust the range. This node allows for precise color-based mask creation, ideal for tasks like object isolation, background removal, or targeted color adjustments.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {}),
                Lexicon.START: ("VEC3INT", {"default": (128, 128, 128), "rgb": True}),
                Lexicon.BOOLEAN: ("BOOLEAN", {"default": False, "tooltip": "use an end point (start->end) when calculating the filter range"}),
                Lexicon.END: ("VEC3INT", {"default": (128, 128, 128), "rgb": True}),
                Lexicon.FLOAT: ("VEC3", {"default": (0.5,0.5,0.5), "min":0, "max":1, "tooltip": "the fuzziness use to extend the start and end range(s)"}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[Any, ...]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        start = parse_param(kw, Lexicon.START, EnumConvertType.VEC3INT, [(128,128,128)], 0, 255)
        use_range = parse_param(kw, Lexicon.BOOLEAN, EnumConvertType.VEC3, [(0,0,0)], 0, 255)
        end = parse_param(kw, Lexicon.END, EnumConvertType.VEC3INT, [(128,128,128)], 0, 255)
        fuzz = parse_param(kw, Lexicon.FLOAT, EnumConvertType.VEC3, [(0.5,0.5,0.5)], 0, 1)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, start, use_range, end, fuzz, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, start, use_range, end, fuzz, matte) in enumerate(params):
            img = np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8) if pA is None else tensor2cv(pA)

            img, mask = image_filter(img, start, end, fuzz, use_range)
            if img.shape[2] == 3:
                alpha_channel = np.zeros((img.shape[0], img.shape[1], 1), dtype=img.dtype)
                img = np.concatenate((img, alpha_channel), axis=2)
            img[..., 3] = mask[:,:]
            images.append(cv2tensor_full(img, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class Flatten(JOVImageNode):
    NAME = "FLATTEN (JOV) ⬇️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 500
    DESCRIPTION = """
Combine multiple input images into a single image by summing their pixel values. This operation is useful for merging multiple layers or images into one composite image, such as combining different elements of a design or merging masks. Users can specify the blending mode and interpolation method to control how the images are combined. Additionally, a matte can be applied to adjust the transparency of the final composite image.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "min":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> torch.Tensor:
        imgs = parse_dynamic(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        pA = []
        for img in imgs:
            pA.extend([image_convert(tensor2cv(i), 4) for i in img])
        if len(pA) == 0:
            logger.error("no images to flatten")
            return ()
        mode = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, EnumScaleMode.NONE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumConvertType.STRING, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        images = []
        params = list(zip_longest_fill(mode, sample, wihi, matte))
        pbar = ProgressBar(len(params))
        for idx, (mode, sample, wihi, matte) in enumerate(params):
            mode = EnumScaleMode[mode]
            h, w = pA[0].shape[:2] if mode == EnumScaleMode.NONE else wihi[::-1]
            current = np.full((w, h, 4), (0,0,0,0), dtype=np.uint8)
            for x in pA:
                if mode != EnumScaleMode.NONE:
                    x = image_scalefit(x, w, h, mode, sample)
                x = image_scalefit(x, w, h, EnumScaleMode.CROP, sample)
                x = image_convert(x, 4)
                #@TODO: ADD VARIOUS COMP OPS?
                current = cv2.add(current, x)
            images.append(cv2tensor_full(current, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class GradientMap(JOVImageNode):
    NAME = "GRADIENT MAP (JOV) 🇲🇺"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 550
    DESCRIPTION = """
Remaps an input image using a gradient lookup table (LUT). The gradient image will be translated into a single row lookup table.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {"tooltip":"Image to remap with gradient input"}),
                Lexicon.GRADIENT: (JOV_TYPE_IMAGE, {"tooltip":f"Look up table (LUT) to remap the input image in `{Lexicon.PIXEL}`"}),
                Lexicon.FLIP: ("BOOLEAN", {"default":False, "tooltip":"Reverse the gradient from left-to-right "}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "min":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> torch.Tensor:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        gradient = parse_param(kw, Lexicon.GRADIENT, EnumConvertType.IMAGE, None)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, EnumScaleMode.NONE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumConvertType.STRING, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        images = []
        params = list(zip_longest_fill(pA, gradient, flip, mode, sample, wihi, matte))
        pbar = ProgressBar(len(params))
        for idx, (pA, gradient, flip, mode, sample, wihi, matte) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor2cv(pA)
            gradient = channel_solid(chan=EnumImageType.BGRA) if gradient is None else tensor2cv(gradient)
            pA = image_gradient_map(pA, gradient)
            # @TODO: pattern o' scale... when make it a lambda?
            mode = EnumScaleMode[mode]
            if mode != EnumScaleMode.NONE:
                w, h = wihi
                sample = EnumInterpolation[sample]
                pA = image_scalefit(pA, w, h, mode, sample)
            #
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class PixelMergeNode(JOVImageNode):
    NAME = "PIXEL MERGE (JOV) 🫂"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 45
    DESCRIPTION = """
Combines individual color channels (red, green, blue) along with an optional mask channel to create a composite image. This node is useful for merging separate color components into a single image for visualization or further processing.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.R: (JOV_TYPE_IMAGE, {}),
                Lexicon.G: (JOV_TYPE_IMAGE, {}),
                Lexicon.B: (JOV_TYPE_IMAGE, {}),
                Lexicon.A: (JOV_TYPE_IMAGE, {}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "min":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
                Lexicon.FLIP: ("VEC4", {"min":0, "max":1, "tooltip": "Invert specific input prior to merging. R, G, B, A."}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the final merged output"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, torch.Tensor]:
        R = parse_param(kw, Lexicon.R, EnumConvertType.MASK, None)
        G = parse_param(kw, Lexicon.G, EnumConvertType.MASK, None)
        B = parse_param(kw, Lexicon.B, EnumConvertType.MASK, None)
        A = parse_param(kw, Lexicon.A, EnumConvertType.MASK, None)
        if len(R)+len(B)+len(G)+len(A) == 0:
            img = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 0, EnumImageType.BGRA)
            return list(cv2tensor_full(img, matte))
        mode = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, EnumScaleMode.NONE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumConvertType.STRING, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.VEC4, [(0, 0, 0, 0)], 0., 1.)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(R, G, B, A, mode, wihi, sample, matte, flip, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (r, g, b, a, mode, wihi, sample, matte, flip, invert) in enumerate(params):
            img = [None if x is None else tensor2cv(x) for x in (r,g,b,a)]
            _, _, w_max, h_max = image_minmax(img)
            for i, x in enumerate(img):
                img[i] = x
                if x is None:
                    # full channel with chosen "level" of color
                    img[i] = np.full((h_max, w_max), int(flip[i] * 255.), dtype=np.uint8)
                elif flip[i] > 0:
                    img[i] = image_invert(img[i], flip[i])

            img = channel_merge(img)
            mode = EnumScaleMode[mode]
            if mode != EnumScaleMode.NONE:
                w, h = wihi
                sample = EnumInterpolation[sample]
                img = image_scalefit(img, w, h, mode, sample)
            if invert == True:
                img = image_invert(img, 1)
            images.append(cv2tensor_full(img, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class PixelSplitNode(JOVBaseNode):
    NAME = "PIXEL SPLIT (JOV) 💔"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK",)
    RETURN_NAMES = (Lexicon.RI, Lexicon.GI, Lexicon.BI, Lexicon.MI)
    SORT = 40
    DESCRIPTION = """
Takes an input image and splits it into its individual color channels (red, green, blue), along with a mask channel. This node is useful for separating different color components of an image for further processing or analysis.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        images = []
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        pbar = ProgressBar(len(pA))
        for idx, pA in enumerate(pA):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor2cv(pA)
            pA = image_mask_add(pA)
            pA = [cv2tensor(x, True) for x in image_split(pA)]
            images.append(pA)
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class PixelSwapNode(JOVImageNode):
    NAME = "PIXEL SWAP (JOV) 🔃"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 48
    DESCRIPTION = """
Swap pixel values between two input images based on specified channel swizzle operations. Options include pixel inputs, swap operations for red, green, blue, and alpha channels, and constant values for each channel. The swap operations allow for flexible pixel manipulation by determining the source of each channel in the output image, whether it be from the first image, the second image, or a constant value.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {}),
                Lexicon.PIXEL_B: (JOV_TYPE_IMAGE, {}),
                Lexicon.SWAP_R: (EnumPixelSwizzle._member_names_,
                                {"default": EnumPixelSwizzle.RED_A.name}),
                Lexicon.R: ("INT", {"default": 0, "min": 0, "max": 255}),
                Lexicon.SWAP_G: (EnumPixelSwizzle._member_names_,
                                {"default": EnumPixelSwizzle.GREEN_A.name}),
                Lexicon.G: ("INT", {"default": 0, "min": 0, "max": 255}),
                Lexicon.SWAP_B: (EnumPixelSwizzle._member_names_,
                                {"default": EnumPixelSwizzle.BLUE_A.name}),
                Lexicon.B: ("INT", {"default": 0, "min": 0, "max": 255}),
                Lexicon.SWAP_A: (EnumPixelSwizzle._member_names_,
                                {"default": EnumPixelSwizzle.ALPHA_A.name}),
                Lexicon.A: ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        swap_r = parse_param(kw, Lexicon.SWAP_R, EnumConvertType.STRING, EnumPixelSwizzle.RED_A.name)
        r = parse_param(kw, Lexicon.R, EnumConvertType.INT, 0, 0, 255)
        swap_g = parse_param(kw, Lexicon.SWAP_G, EnumConvertType.STRING, EnumPixelSwizzle.GREEN_A.name)
        g = parse_param(kw, Lexicon.G, EnumConvertType.INT, 0, 0, 255)
        swap_b = parse_param(kw, Lexicon.SWAP_B, EnumConvertType.STRING, EnumPixelSwizzle.BLUE_A.name)
        b = parse_param(kw, Lexicon.B, EnumConvertType.INT, 0, 0, 255)
        swap_a = parse_param(kw, Lexicon.SWAP_A, EnumConvertType.STRING, EnumPixelSwizzle.ALPHA_A.name)
        a = parse_param(kw, Lexicon.A, EnumConvertType.INT, 0, 0, 255)
        params = list(zip_longest_fill(pA, pB, r, swap_r, g, swap_g, b, swap_b, a, swap_a))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, r, swap_r, g, swap_g, b, swap_b, a, swap_a) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            h, w = pA.shape[:2]
            pB = tensor2cv(pB) if pB is not None else channel_solid(w, h, chan=EnumImageType.BGRA)
            out = channel_solid(w, h, (r,g,b,a), EnumImageType.BGRA)

            if len(pA) < 2 or pA.shape[2] < 4:
                pA = image_convert(pA, 4)
            if len(pB) < 2 or pB.shape[2] < 4:
                pB = image_convert(pB, 4)

            # crop fit?
            pB = image_scalefit(pB, w, h, EnumScaleMode.CROP)

            def swapper(swap_out:EnumPixelSwizzle, swap_in:EnumPixelSwizzle) -> np.ndarray[Any]:
                target = out
                swap_in = EnumPixelSwizzle[swap_in]
                if swap_in in [EnumPixelSwizzle.RED_A, EnumPixelSwizzle.GREEN_A,
                            EnumPixelSwizzle.BLUE_A, EnumPixelSwizzle.ALPHA_A]:
                    target = pA
                elif swap_in in [EnumPixelSwizzle.RED_B, EnumPixelSwizzle.GREEN_B,
                            EnumPixelSwizzle.BLUE_B, EnumPixelSwizzle.ALPHA_B]:
                    target = pB
                elif swap_in != EnumPixelSwizzle.CONSTANT:
                    target = channel_swap(pA, swap_out, pB, swap_in)
                return target

            # logger.debug(swap_r, swap_g, swap_b, swap_a)
            out[..., 0] = swapper(EnumPixelSwizzle.BLUE_A, swap_b)[..., 0]
            out[:,:,1] = swapper(EnumPixelSwizzle.GREEN_A, swap_g)[:,:,1]
            out[:,:,2] = swapper(EnumPixelSwizzle.RED_A, swap_r)[:,:,2]
            out[..., 3] = swapper(EnumPixelSwizzle.ALPHA_A, swap_a)[..., 3]
            images.append(cv2tensor_full(out))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class StackNode(JOVImageNode):
    NAME = "STACK (JOV) ➕"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 75
    DESCRIPTION = """
Merge multiple input images into a single composite image by stacking them along a specified axis. Options include axis, stride, scaling mode, width and height, interpolation method, and matte color. The axis parameter allows for horizontal, vertical, or grid stacking of images, while stride controls the spacing between them.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.AXIS: (EnumOrientation._member_names_, {"default": EnumOrientation.GRID.name,
                                                                "tooltip":"Choose the direction in which to stack the images. Options include horizontal, vertical, or a grid layout"}),
                Lexicon.STEP: ("INT", {"min": 0, "default": 1,
                                    "tooltip":"Specify the spacing between each stacked image. This determines how far apart the images are from each other"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "min":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        images = parse_dynamic(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        if len(images) == 0:
            logger.warning("no images to stack")
            return
        data = []
        for i in images:
            data.extend(i)
        images = [tensor2cv(i) for i in data]

        axis = parse_param(kw, Lexicon.AXIS, EnumConvertType.STRING, EnumOrientation.GRID.name)[0]
        stride = parse_param(kw, Lexicon.STEP, EnumConvertType.INT, 1)[0]
        mode = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, EnumScaleMode.NONE.name)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)[0]
        sample = parse_param(kw, Lexicon.SAMPLE, EnumConvertType.STRING, EnumInterpolation.LANCZOS4.name)[0]
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)[0]
        axis = EnumOrientation[axis]
        img = image_stack(images, axis, stride) #, matte)
        mode = EnumScaleMode[mode]
        if mode != EnumScaleMode.NONE:
            w, h = wihi
            sample = EnumInterpolation[sample]
            img = image_scalefit(img, w, h, mode, sample)
        return cv2tensor_full(img, matte)

class ThresholdNode(JOVImageNode):
    NAME = "THRESHOLD (JOV) 📉"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Define a range and apply it to an image for segmentation and feature extraction. Choose from various threshold modes, such as binary and adaptive, and adjust the threshold value and block size to suit your needs. You can also invert the resulting mask if necessary. This node is versatile for a variety of image processing tasks.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.ADAPT: ( EnumThresholdAdapt._member_names_,
                                {"default": EnumThresholdAdapt.ADAPT_NONE.name}),
                Lexicon.FUNC: ( EnumThreshold._member_names_, {"default": EnumThreshold.BINARY.name}),
                Lexicon.THRESHOLD: ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.005}),
                Lexicon.SIZE: ("INT", {"default": 3, "min": 3, "max": 103}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        mode = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumThreshold.BINARY.name)
        adapt = parse_param(kw, Lexicon.ADAPT, EnumConvertType.STRING, EnumThresholdAdapt.ADAPT_NONE.name)
        threshold = parse_param(kw, Lexicon.THRESHOLD, EnumConvertType.FLOAT, 1, 0, 1)
        block = parse_param(kw, Lexicon.SIZE, EnumConvertType.INT, 3, 3)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, mode, adapt, threshold, block, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mode, adapt, th, block, invert) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            mode = EnumThreshold[mode]
            adapt = EnumThresholdAdapt[adapt]
            pA = image_threshold(pA, th, mode, adapt, block)
            if invert == True:
                pA = image_invert(pA, 1)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class TransformNode(JOVImageNode):
    NAME = "TRANSFORM (JOV) 🏝️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    SORT = 0
    DESCRIPTION = """
Apply various geometric transformations to images, including translation, rotation, scaling, mirroring, tiling and perspective projection. It offers extensive control over image manipulation to achieve desired visual effects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.XY: ("VEC2", {"default": (0, 0,), "min": -1, "max": 1, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.ANGLE: ("FLOAT", {"default": 0}),
                Lexicon.SIZE: ("VEC2", {"default": (1., 1.), "min": 0.001, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.TILE: ("VEC2", {"default": (1., 1.), "min": 1, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
                Lexicon.MIRROR: (EnumMirrorMode._member_names_, {"default": EnumMirrorMode.NONE.name}),
                Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.005, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.PROJECTION: (EnumProjection._member_names_, {"default": EnumProjection.NORMAL.name}),
                Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "step": 0.005,  "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
                Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "step": 0.005, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
                Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "step": 0.005}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "min":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        offset = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, [(0, 0)], -1, 1)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        size = parse_param(kw, Lexicon.SIZE, EnumConvertType.VEC2, [(1, 1)], 0.001)
        edge = parse_param(kw, Lexicon.EDGE, EnumConvertType.STRING, EnumEdge.CLIP.name)
        mirror = parse_param(kw, Lexicon.MIRROR, EnumConvertType.STRING, EnumMirrorMode.NONE.name)
        mirror_pivot = parse_param(kw, Lexicon.PIVOT, EnumConvertType.VEC2, [(0.5, 0.5)], 0, 1)
        tile_xy = parse_param(kw, Lexicon.TILE, EnumConvertType.VEC2, [(1., 1.)], 1)
        proj = parse_param(kw, Lexicon.PROJECTION, EnumConvertType.STRING, EnumProjection.NORMAL.name)
        tltr = parse_param(kw, Lexicon.TLTR, EnumConvertType.VEC4, [(0, 0, 1, 0)], 0, 1)
        blbr = parse_param(kw, Lexicon.BLBR, EnumConvertType.VEC4, [(0, 1, 1, 1)], 0, 1)
        strength = parse_param(kw, Lexicon.STRENGTH, EnumConvertType.FLOAT, 1, 0, 1)
        mode = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, EnumScaleMode.NONE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumConvertType.STRING, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, matte) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            h, w = pA.shape[:2]
            edge = EnumEdge[edge]
            sample = EnumInterpolation[sample]
            pA = image_transform(pA, offset, angle, size, sample, edge)
            pA = image_crop_center(pA, w, h)

            mirror = EnumMirrorMode[mirror]
            if mirror != EnumMirrorMode.NONE:
                mpx, mpy = mirror_pivot
                pA = image_mirror(pA, mirror, mpx, mpy)
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample)

            tx, ty = tile_xy
            if tx != 1. or ty != 1.:
                pA = image_edge_wrap(pA, tx / 2 - 0.5, ty / 2 - 0.5)
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample)

            proj = EnumProjection[proj]
            match proj:
                case EnumProjection.PERSPECTIVE:
                    x1, y1, x2, y2 = tltr
                    x4, y4, x3, y3 = blbr
                    sh, sw = pA.shape[:2]
                    x1, x2, x3, x4 = map(lambda x: x * sw, [x1, x2, x3, x4])
                    y1, y2, y3, y4 = map(lambda y: y * sh, [y1, y2, y3, y4])
                    pA = remap_perspective(pA, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                case EnumProjection.SPHERICAL:
                    pA = remap_sphere(pA, strength)
                case EnumProjection.FISHEYE:
                    pA = remap_fisheye(pA, strength)
                case EnumProjection.POLAR:
                    pA = remap_polar(pA)

            if proj != EnumProjection.NORMAL:
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample)

            mode = EnumScaleMode[mode]
            if mode != EnumScaleMode.NONE:
                w, h = wihi
                pA = image_scalefit(pA, w, h, mode, sample)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

'''
class HistogramNode(JOVImageSimple):
    NAME = "HISTOGRAM (JOV) 👁‍🗨"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE,)
    SORT = 40
    DESCRIPTION = """
The Histogram Node generates a histogram representation of the input image, showing the distribution of pixel intensity values across different bins. This visualization is useful for understanding the overall brightness and contrast characteristics of an image. Additionally, the node performs histogram normalization, which adjusts the pixel values to enhance the contrast of the image. Histogram normalization can be helpful for improving the visual quality of images or preparing them for further image processing tasks.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, None), EnumConvertType.IMAGE, None)
        params = list(zip_longest_fill(pA,))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, ) in enumerate(params):
            pA = image_histogram(pA)
            pA = image_histogram_normalize(pA)
            images.append(cv2tensor(pA))
            pbar.update_absolute(idx)
        return list(zip(*images))
'''
