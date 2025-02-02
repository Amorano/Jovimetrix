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

from .. import JOV_TYPE_IMAGE, \
    JOVBaseNode, JOVImageNode, Lexicon, \
    deep_merge

from ..sup.util import EnumConvertType, \
    parse_dynamic, parse_param, zip_longest_fill

from ..sup.image import MIN_IMAGE_SIZE, \
    EnumImageType, \
    image_mask, image_mask_add, image_matte, image_minmax, image_convert, \
    cv2tensor, cv2tensor_full, tensor2cv

from ..sup.image.color import EnumCBDeficiency, EnumCBSimulator, EnumColorMap, EnumColorTheory, \
    color_lut_full, color_lut_match, color_lut_palette, \
    color_lut_tonal, color_lut_visualize, color_match_reinhard, color_theory, color_blind, \
    color_top_used, image_gradient_expand, image_gradient_map, pixel_eval

from ..sup.image.adjust import EnumEdge, EnumMirrorMode, EnumScaleMode, \
    EnumInterpolation, EnumThreshold, EnumThresholdAdapt, \
    image_contrast, image_edge_wrap, image_equalize, image_filter, image_gamma,  \
    image_hsv, image_invert, image_mirror, image_pixelate, image_posterize, \
    image_quantize, image_scalefit, image_sharpen, image_swap_channels, \
    image_transform, image_flatten, image_threshold, morph_edge_detect, morph_emboss

from ..sup.image.channel import EnumPixelSwizzle, \
    channel_merge, channel_solid

from ..sup.image.compose import EnumAdjustOP, EnumBlendType, EnumOrientation, \
    image_levels, image_split, image_stack, image_blend, \
    image_crop, image_crop_center, image_crop_polygonal

from ..sup.image.mapping import EnumProjection, \
    remap_fisheye, remap_perspective, remap_polar, remap_sphere

# ==============================================================================

JOV_CATEGORY = "COMPOSE"

class EnumColorMatchMode(Enum):
    REINHARD = 30
    LUT = 10
    # HISTOGRAM = 20

class EnumColorMatchMap(Enum):
    USER_MAP = 0
    PRESET_MAP = 10

class EnumCropMode(Enum):
    CENTER = 20
    XY = 0
    FREE = 10
    HEAD = 15
    BODY = 25

# ==============================================================================

class AdjustNode(JOVImageNode):
    NAME = "ADJUST (JOV) 🕸️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Enhance and modify images with various effects such as blurring, sharpening, color tweaks, and edge detection. Customize parameters like radius, value, and contrast, and use masks for selective effects. Advanced options include pixelation, quantization, and morphological operations like dilation and erosion. Handle transparency effortlessly to ensure seamless blending of effects. This node is ideal for simple adjustments and complex image transformations.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.MASK: (JOV_TYPE_IMAGE, {}),
                Lexicon.FUNC: (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name,
                                                            "tooltip":"Type of adjustment (e.g., blur, sharpen, invert)"}),
                Lexicon.RADIUS: ("INT", {"default": 3, "min": 3}),
                Lexicon.VALUE: ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
                Lexicon.LOHI: ("VEC2", {"default": (0, 1),
                                        "mij": 0, "maj": 1, "step": 0.01, "label": [Lexicon.LO, Lexicon.HI]}),
                Lexicon.LMH: ("VEC3", {"default": (0, 0.5, 1),
                                        "mij": 0, "maj": 1, "step": 0.01, "label": [Lexicon.LO, Lexicon.MID, Lexicon.HI]}),
                Lexicon.HSV: ("VEC3",{"default": (0, 1, 1),
                                    "mij": 0, "maj": 1, "step": 0.01,  "label": [Lexicon.H, Lexicon.S, Lexicon.V]}),
                Lexicon.CONTRAST: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.GAMMA: ("FLOAT", {"default": 1, "min": 0.00001, "max": 1, "step": 0.01}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, ...]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNC, EnumAdjustOP, EnumAdjustOP.BLUR.name)
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
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGR)
            img_new = image_convert(pA, 3)

            match op:
                case EnumAdjustOP.INVERT:
                    img_new = image_invert(img_new, val)

                case EnumAdjustOP.LEVELS:
                    l, m, h = lmh
                    img_new = image_levels(img_new, l, h, m, gamma)

                case EnumAdjustOP.HSV:
                    h, s, v = hsv
                    img_new = image_hsv(img_new, h, s, v)
                    if contrast != 0:
                        img_new = image_contrast(img_new, 1 - contrast)

                    if gamma != 0:
                        img_new = image_gamma(img_new, gamma)

                case EnumAdjustOP.FIND_EDGES:
                    lo, hi = lohi
                    img_new = morph_edge_detect(img_new, low=lo, high=hi)

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

                case EnumAdjustOP.EQUALIZE:
                    img_new = image_equalize(img_new)

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

            if mask is not None:
                mask = tensor2cv(mask)
                if invert:
                    mask = 255 - mask

            img_new = image_blend(pA, img_new, mask)
            if pA.ndim == 3 and pA.shape[2] == 4:
                mask = image_mask(pA)
                img_new = image_convert(img_new, 4)
                img_new[:,:,3] = mask
            #    img_new = image_mask_add(mask)

            images.append(cv2tensor_full(img_new, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {"tooltip": "Background Plate"}),
                Lexicon.PIXEL_B: (JOV_TYPE_IMAGE, {"tooltip": "Image to Overlay on Background Plate"}),
                Lexicon.MASK: (JOV_TYPE_IMAGE, {"tooltip": "Optional Mask to use for Alpha Blend Operation. If empty, will use the ALPHA of B"}),
                Lexicon.FUNC: (EnumBlendType._member_names_, {"default": EnumBlendType.NORMAL.name, "tooltip": "Blending Operation"}),
                Lexicon.A: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Amount of Blending to Perform on the Selected Operation"}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        func = parse_param(kw, Lexicon.FUNC, EnumBlendType, EnumBlendType.NORMAL.name)
        alpha = parse_param(kw, Lexicon.A, EnumConvertType.FLOAT, 1, 0, 1)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, pB, mask, func, alpha, flip, mode, wihi, sample, matte, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, mask, func, alpha, flip, mode, wihi, sample, matte, invert) in enumerate(params):
            if flip:
                pA, pB = pB, pA

            width, height = MIN_IMAGE_SIZE, MIN_IMAGE_SIZE
            if pA is None:
                if pB is None:
                    if mask is None:
                        if mode != EnumScaleMode.MATTE:
                            width, height = wihi
                    else:
                        height, width = mask.shape[:2]
                else:
                    height, width = pB.shape[:2]
            else:
                height, width = pA.shape[:2]

            if pA is None:
                pA = channel_solid(width, height, matte, chan=EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)
                matted = pixel_eval(matte, EnumImageType.BGRA)
                pA = image_matte(pA, matted)

            if pB is None:
                pB = channel_solid(width, height, matte, chan=EnumImageType.BGRA)
            else:
                pB = tensor2cv(pB)

            if mask is not None:
                mask = tensor2cv(mask)
                # mask = image_grayscale(mask)
                if invert:
                    mask = 255 - mask

            img = image_blend(pA, pB, mask, func, alpha)

            if mode != EnumScaleMode.MATTE:
                # or mode != EnumScaleMode.RESIZE_MATTE:
                width, height = wihi
                img = image_scalefit(img, width, height, mode, sample)

            img = cv2tensor_full(img, matte)
            images.append(img)
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

class ColorBlindNode(JOVImageNode):
    NAME = "COLOR BLIND (JOV) 👁‍🗨"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Simulate color blindness effects on images. You can select various types of color deficiencies, adjust the severity of the effect, and apply the simulation using different simulators. This node is ideal for accessibility testing and design adjustments, ensuring inclusivity in your visual content.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
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
        deficiency = parse_param(kw, Lexicon.DEFICIENCY, EnumCBDeficiency, EnumCBDeficiency.PROTAN.name)
        simulator = parse_param(kw, Lexicon.SIMULATOR, EnumCBSimulator, EnumCBSimulator.AUTOSELECT.name)
        severity = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 1)
        params = list(zip_longest_fill(pA, deficiency, simulator, severity))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, deficiency, simulator, severity) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor2cv(pA)
            pA = color_blind(pA, deficiency, simulator, severity)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

class ColorMatchNode(JOVImageNode):
    NAME = "COLOR MATCH (JOV) 💞"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Adjust the color scheme of one image to match another with the Color Match Node. Choose from various color matching LUTs or Reinhard matching. You can specify a custom user color maps, the number of colors, and whether to flip or invert the images.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {}),
                Lexicon.PIXEL_B: (JOV_TYPE_IMAGE, {}),
                Lexicon.COLORMATCH_MODE: (EnumColorMatchMode._member_names_,
                                            {"default": EnumColorMatchMode.REINHARD.name}),
                Lexicon.COLORMATCH_MAP: (EnumColorMatchMap._member_names_,
                                            {"default": EnumColorMatchMap.USER_MAP.name}),
                Lexicon.COLORMAP: (EnumColorMap._member_names_,
                                    {"default": EnumColorMap.HSV.name}),
                Lexicon.VALUE: ("INT", {"default": 255, "min": 0, "max": 255, "tooltip":"The number of colors to use from the LUT during the remap. Will quantize the LUT range."}),
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
        colormatch_mode = parse_param(kw, Lexicon.COLORMATCH_MODE, EnumColorMatchMode, EnumColorMatchMode.REINHARD.name)
        colormatch_map = parse_param(kw, Lexicon.COLORMATCH_MAP, EnumColorMatchMap, EnumColorMatchMap.USER_MAP.name)
        colormap = parse_param(kw, Lexicon.COLORMAP, EnumColorMap, EnumColorMap.HSV.name)
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

            mask = None
            if pA is None:
                pA = channel_solid(chan=EnumImageType.BGR)
            else:
                pA = tensor2cv(pA)
                if pA.ndim == 3 and pA.shape[2] == 4:
                    mask = image_mask(pA)

            # h, w = pA.shape[:2]
            if pB is None:
                pB = channel_solid(chan=EnumImageType.BGR)
            else:
                pB = tensor2cv(pB)

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

            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

class ColorKMeansNode(JOVBaseNode):
    NAME = "COLOR MEANS (JOV) 〰️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "JLUT", "IMAGE",)
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.PALETTE, Lexicon.GRADIENT, Lexicon.LUT, Lexicon.RGB, )
    DESCRIPTION = """
The top-k colors ordered from most->least used as a strip, tonal palette and 3D LUT.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.VALUE: ("INT", {"default": 12, "min": 1, "max": 255, "tooltip":"The top K colors to select."}),
                Lexicon.SIZE: ("INT", {"default": 32, "min": 1, "max": 256, "tooltip":"Height of the tones in the strip. Width is based on input."}),
                Lexicon.COUNT: ("INT", {"default": 33, "min": 3, "max": 256, "tooltip":"Number of nodes to use in interpolation of full LUT (256 is every pixel)."}),
                Lexicon.WH: ("VEC2INT", {"default": (256, 256), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
            },
            "outputs": {
                0: ("IMAGE", {"tooltip":"Sequence of top-K colors. Count depends on value in `VAL`."}),
                1: ("IMAGE", {"tooltip":"Simple Tone palette based on result top-K colors. Width is taken from input."}),
                2: ("IMAGE", {"tooltip":"Gradient of top-K colors."}),
                3: ("JLUT",  {"tooltip":"Full 3D LUT of the image mapped to the resultant top-K colors chosen."}),
                4: ("IMAGE", {"tooltip":"Visualization of full 3D .cube LUT in JLUT output"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        kcolors = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 12, 1, 255)
        lut_height = parse_param(kw, Lexicon.SIZE, EnumConvertType.INT, 32, 1, 256)
        nodes = parse_param(kw, Lexicon.COUNT, EnumConvertType.INT, 33, 1, 255)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(256, 256)], MIN_IMAGE_SIZE)

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

            pA = tensor2cv(pA)
            colors = color_top_used(pA, kcolors)

            # size down to 1px strip then expand to 256 for full gradient
            top_colors.extend([cv2tensor(channel_solid(*wihi, color=c)) for c in colors])
            lut_tonal.append(cv2tensor(color_lut_tonal(colors, width=pA.shape[1], height=lut_height)))
            full = color_lut_full(colors, nodes)
            lut_full.append(torch.from_numpy(full))
            lut_visualized.append(cv2tensor(color_lut_visualize(full, wihi[1])))
            gradient = image_gradient_expand(color_lut_palette(colors, 1))
            gradient = cv2.resize(gradient, wihi)
            gradients.append(cv2tensor(gradient))
            pbar.update_absolute(idx)

        return torch.stack(top_colors), torch.stack(lut_tonal), torch.stack(gradients), lut_full, torch.stack(lut_visualized),

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.SCHEME: (EnumColorTheory._member_names_, {"default": EnumColorTheory.COMPLIMENTARY.name}),
                Lexicon.VALUE: ("INT", {"default": 45, "min": -90, "max": 90,
                                        "tooltip": "Custom angle of separation to use when calculating colors"}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        scheme = parse_param(kw, Lexicon.SCHEME, EnumColorTheory, EnumColorTheory.COMPLIMENTARY.name)
        user = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 0, -180, 180)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, scheme, user, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (img, target, user, invert) in enumerate(params):
            img = tensor2cv(img) if img is not None else channel_solid(chan=EnumImageType.BGRA)
            img = color_theory(img, user, target)
            if invert:
                img = (image_invert(s, 1) for s in img)
            images.append([cv2tensor(a) for a in img])
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.FUNC: (EnumCropMode._member_names_, {"default": EnumCropMode.CENTER.name}),
                Lexicon.XY: ("VEC2", {"default": (0, 0), "mij": 0.5, "maj": 0.5, "step": 0.01, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij": MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.TLTR: ("VEC4", {"default": (0, 0, 0, 1), "mij": 0, "maj": 1, "step": 0.01, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
                Lexicon.BLBR: ("VEC4", {"default": (1, 0, 1, 1), "mij": 0, "maj": 1, "step": 0.01,  "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        func = parse_param(kw, Lexicon.FUNC, EnumCropMode, EnumCropMode.CENTER.name)
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
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {}),
                Lexicon.START: ("VEC3INT", {"default": (128, 128, 128), "rgb": True}),
                Lexicon.BOOLEAN: ("BOOLEAN", {"default": False, "tooltip": "use an end point (start->end) when calculating the filter range"}),
                Lexicon.END: ("VEC3INT", {"default": (128, 128, 128), "rgb": True}),
                Lexicon.FLOAT: ("VEC3", {"default": (0.5,0.5,0.5), "mij":0, "maj":1, "step": 0.01, "tooltip": "the fuzziness use to extend the start and end range(s)"}),
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
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> torch.Tensor:
        imgs = parse_dynamic(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        if imgs is None:
            logger.error("no images to flatten")
            return ()

        # be less dumb when merging
        pA = [tensor2cv(i) for img in imgs for i in img]
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)

        images = []
        params = list(zip_longest_fill(mode, sample, wihi, matte))
        pbar = ProgressBar(len(params))
        for idx, (mode, sample, wihi, matte) in enumerate(params):
            current = image_flatten(pA)
            images.append(cv2tensor_full(current, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {"tooltip":"Image to remap with gradient input"}),
                Lexicon.GRADIENT: (JOV_TYPE_IMAGE, {"tooltip":f"Look up table (LUT) to remap the input image in `{Lexicon.PIXEL}`"}),
                Lexicon.FLIP: ("BOOLEAN", {"default":False, "tooltip":"Reverse the gradient from left-to-right "}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> torch.Tensor:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        gradient = parse_param(kw, Lexicon.GRADIENT, EnumConvertType.IMAGE, None)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        images = []
        params = list(zip_longest_fill(pA, gradient, flip, mode, sample, wihi, matte))
        pbar = ProgressBar(len(params))
        for idx, (pA, gradient, flip, mode, sample, wihi, matte) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGR) if pA is None else tensor2cv(pA)
            mask = None
            if pA.ndim == 3 and pA.shape[2] == 4:
                mask = image_mask(pA)

            gradient = channel_solid(chan=EnumImageType.BGR) if gradient is None else tensor2cv(gradient)
            pA = image_gradient_map(pA, gradient)
            if mode != EnumScaleMode.MATTE:
                w, h = wihi
                pA = image_scalefit(pA, w, h, mode, sample)
            if mask is not None:
                pA = image_mask_add(pA, mask)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.R: (JOV_TYPE_IMAGE, {}),
                Lexicon.G: (JOV_TYPE_IMAGE, {}),
                Lexicon.B: (JOV_TYPE_IMAGE, {}),
                Lexicon.A: (JOV_TYPE_IMAGE, {}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
                Lexicon.FLIP: ("VEC4", {"mij":0, "maj":1, "step": 0.01, "tooltip": "Invert specific input prior to merging. R, G, B, A."}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the final merged output"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, torch.Tensor]:
        rgba = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        R = parse_param(kw, Lexicon.R, EnumConvertType.MASK, None)
        G = parse_param(kw, Lexicon.G, EnumConvertType.MASK, None)
        B = parse_param(kw, Lexicon.B, EnumConvertType.MASK, None)
        A = parse_param(kw, Lexicon.A, EnumConvertType.MASK, None)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.VEC4, [(0, 0, 0, 0)], 0., 1.)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(rgba, R, G, B, A, mode, wihi, sample, matte, flip, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (rgba, r, g, b, a, mode, wihi, sample, matte, flip, invert) in enumerate(params):
            replace = r, g, b, a
            if rgba is not None:
                rgba = tensor2cv(rgba)
                rgba = image_convert(rgba, 4)
                rgba = image_split(rgba)
                img = [tensor2cv(replace[i]) if replace[i] is not None else x for i, x in enumerate(rgba)]
            else:
                img = [tensor2cv(x) if x is not None else x for x in replace]

            _, _, w_max, h_max = image_minmax(img)
            for i, x in enumerate(img):
                img[i] = x
                if x is None:
                    img[i] = np.full((h_max, w_max), matte[i], dtype=np.uint8)
                if flip[i] > 0:
                    img[i] = image_invert(x, flip[i])

            img = channel_merge(img)

            if mode != EnumScaleMode.MATTE:
                w, h = wihi
                img = image_scalefit(img, w, h, mode, sample)

            if invert == True:
                img = image_invert(img, 1)

            images.append(cv2tensor_full(img, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {})
            },
            "outputs": {
                0: ("MASK", {"tooltip":"Single channel output of Red Channel."}),
                1: ("MASK", {"tooltip":"Single channel output of Green Channel"}),
                2: ("MASK", {"tooltip":"Single channel output of Blue Channel"}),
                3: ("MASK", {"tooltip":"Single channel output of Alpha Channel"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        images = []
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        pbar = ProgressBar(len(pA))
        for idx, pA in enumerate(pA):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor2cv(pA)
            images.append([cv2tensor(x, True) for x in image_split(pA)])
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (JOV_TYPE_IMAGE, {}),
                Lexicon.PIXEL_B: (JOV_TYPE_IMAGE, {}),
                Lexicon.SWAP_R: (EnumPixelSwizzle._member_names_,
                                {"default": EnumPixelSwizzle.RED_A.name}),
                Lexicon.SWAP_G: (EnumPixelSwizzle._member_names_,
                                {"default": EnumPixelSwizzle.GREEN_A.name}),
                Lexicon.SWAP_B: (EnumPixelSwizzle._member_names_,
                                {"default": EnumPixelSwizzle.BLUE_A.name}),
                Lexicon.SWAP_A: (EnumPixelSwizzle._member_names_,
                                {"default": EnumPixelSwizzle.ALPHA_A.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        swap_r = parse_param(kw, Lexicon.SWAP_R, EnumPixelSwizzle, EnumPixelSwizzle.RED_A.name)
        swap_g = parse_param(kw, Lexicon.SWAP_G, EnumPixelSwizzle, EnumPixelSwizzle.GREEN_A.name)
        swap_b = parse_param(kw, Lexicon.SWAP_B, EnumPixelSwizzle, EnumPixelSwizzle.BLUE_A.name)
        swap_a = parse_param(kw, Lexicon.SWAP_A, EnumPixelSwizzle, EnumPixelSwizzle.ALPHA_A.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, pB, swap_r, swap_g, swap_b, swap_a, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, swap_r, swap_g, swap_b, swap_a, matte) in enumerate(params):
            if pA is None:
                if pB is None:
                    out = channel_solid(chan=EnumImageType.BGRA)
                    images.append(cv2tensor_full(out))
                    pbar.update_absolute(idx)
                    continue

                h, w = pB.shape[:2]
                pA = channel_solid(w, h, chan=EnumImageType.BGRA)
            else:
                h, w = pA.shape[:2]
                pA = tensor2cv(pA)
                pA = image_convert(pA, 4)

            pB = tensor2cv(pB) if pB is not None else channel_solid(w, h, chan=EnumImageType.BGRA)
            pB = image_convert(pB, 4)
            pB = image_matte(pB, (0,0,0,0), w, h)
            pB = image_scalefit(pB, w, h, EnumScaleMode.CROP)

            out = image_swap_channels(pA, pB, (swap_r, swap_g, swap_b, swap_a), matte)

            images.append(cv2tensor_full(out))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.AXIS: (EnumOrientation._member_names_, {"default": EnumOrientation.GRID.name,
                                                                "tooltip":"Choose the direction in which to stack the images. Options include horizontal, vertical, or a grid layout"}),
                Lexicon.STEP: ("INT", {"min": 0, "default": 1,
                                    "tooltip":"Specify the spacing between each stacked image. This determines how far apart the images are from each other"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
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
        images = [tensor2cv(img) for sublist in images for img in sublist]

        axis = parse_param(kw, Lexicon.AXIS, EnumOrientation, EnumOrientation.GRID.name)[0]
        stride = parse_param(kw, Lexicon.STEP, EnumConvertType.INT, 1)[0]
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)[0]
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)[0]
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)[0]
        img = image_stack(images, axis, stride) #, matte)
        if mode != EnumScaleMode.MATTE:
            w, h = wihi
            img = image_scalefit(img, w, h, mode, sample)
        rgba, rgb, mask = cv2tensor_full(img, matte)
        return rgba.unsqueeze(0), rgb.unsqueeze(0), mask.unsqueeze(0)

class ThresholdNode(JOVImageNode):
    NAME = "THRESHOLD (JOV) 📉"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Define a range and apply it to an image for segmentation and feature extraction. Choose from various threshold modes, such as binary and adaptive, and adjust the threshold value and block size to suit your needs. You can also invert the resulting mask if necessary. This node is versatile for a variety of image processing tasks.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
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
        mode = parse_param(kw, Lexicon.FUNC, EnumThreshold, EnumThreshold.BINARY.name)
        adapt = parse_param(kw, Lexicon.ADAPT, EnumThresholdAdapt, EnumThresholdAdapt.ADAPT_NONE.name)
        threshold = parse_param(kw, Lexicon.THRESHOLD, EnumConvertType.FLOAT, 1, 0, 1)
        block = parse_param(kw, Lexicon.SIZE, EnumConvertType.INT, 3, 3)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, mode, adapt, threshold, block, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mode, adapt, th, block, invert) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            pA = image_threshold(pA, th, mode, adapt, block)
            if invert == True:
                pA = image_invert(pA, 1)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.XY: ("VEC2", {"default": (0, 0,), "mij": -1, "maj": 1, "step": 0.01, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.ANGLE: ("FLOAT", {"default": 0, "step": 0.01}),
                Lexicon.SIZE: ("VEC2", {"default": (1., 1.), "mij": 0.001, "step": 0.01, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.TILE: ("VEC2", {"default": (1., 1.), "mij": 1, "step": 0.01, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
                Lexicon.MIRROR: (EnumMirrorMode._member_names_, {"default": EnumMirrorMode.NONE.name}),
                Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.005, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.PROJECTION: (EnumProjection._member_names_, {"default": EnumProjection.NORMAL.name}),
                Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "step": 0.005,  "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
                Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "step": 0.005, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
                Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "step": 0.005}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        offset = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, [(0, 0)], -2.5, 2.5)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        size = parse_param(kw, Lexicon.SIZE, EnumConvertType.VEC2, [(1, 1)], 0.001)
        edge = parse_param(kw, Lexicon.EDGE, EnumEdge, EnumEdge.CLIP.name)
        mirror = parse_param(kw, Lexicon.MIRROR, EnumMirrorMode, EnumMirrorMode.NONE.name)
        mirror_pivot = parse_param(kw, Lexicon.PIVOT, EnumConvertType.VEC2, [(0.5, 0.5)], 0, 1)
        tile_xy = parse_param(kw, Lexicon.TILE, EnumConvertType.VEC2, [(1., 1.)], 1)
        proj = parse_param(kw, Lexicon.PROJECTION, EnumProjection, EnumProjection.NORMAL.name)
        tltr = parse_param(kw, Lexicon.TLTR, EnumConvertType.VEC4, [(0, 0, 1, 0)], 0, 1)
        blbr = parse_param(kw, Lexicon.BLBR, EnumConvertType.VEC4, [(0, 1, 1, 1)], 0, 1)
        strength = parse_param(kw, Lexicon.STRENGTH, EnumConvertType.FLOAT, 1, 0, 1)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, matte) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            h, w = pA.shape[:2]
            pA = image_transform(pA, offset, angle, size, sample, edge)
            pA = image_crop_center(pA, w, h)

            if mirror != EnumMirrorMode.NONE:
                mpx, mpy = mirror_pivot
                pA = image_mirror(pA, mirror, mpx, mpy)
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample)

            tx, ty = tile_xy
            if tx != 1. or ty != 1.:
                pA = image_edge_wrap(pA, tx / 2 - 0.5, ty / 2 - 0.5)
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample)

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

            if mode != EnumScaleMode.MATTE:
                w, h = wihi
                pA = image_scalefit(pA, w, h, mode, sample)

            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

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
        d = deep_merge(d, {
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
