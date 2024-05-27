"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Adjustment
"""

from enum import Enum
from typing import Any, Tuple

import cv2
import torch

from comfy.utils import ProgressBar

from Jovimetrix import JOVBaseNode, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import EnumConvertType, parse_param, zip_longest_fill
from Jovimetrix.sup.image import channel_count, channel_solid, \
    color_match_histogram, color_match_lut, color_match_reinhard, cv2tensor, cv2tensor_full, \
    image_color_blind, image_grayscale, image_scalefit, tensor2cv, image_equalize, \
    image_levels, pixel_eval, image_posterize, image_pixelate, image_quantize, \
    image_sharpen, image_threshold, image_blend, image_invert, morph_edge_detect, \
    morph_emboss, image_contrast, image_hsv, image_gamma, \
    EnumCBDefiency, EnumCBSimulator, EnumScaleMode, \
    EnumImageType, EnumColorMap, EnumAdjustOP, EnumThresholdAdapt, EnumThreshold

# =============================================================================

JOV_CATEGORY = "ADJUST"

class EnumColorMatchMode(Enum):
    REINHARD = 30
    LUT = 10
    HISTOGRAM = 20

class EnumColorMatchMap(Enum):
    USER_MAP = 0
    PRESET_MAP = 10

# =============================================================================

class AdjustNode(JOVBaseNode):
    NAME = "ADJUST (JOV) 🕸️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    DESCRIPTION = """
The `Adjust Node` lets you enhance and modify images with various effects.
You can apply blurring, sharpening, color tweaks, and edge detection.
Customize parameters like radius, value, and contrast, and use masks for
selective effects. Advanced options include pixelation, quantization, and
morphological operations like dilation and erosion. Handle transparency easily,
ensuring seamless blending of effects. Perfect for simple adjustments and
complex image transformations.
"""
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {},
            "optional": {
                Lexicon.PIXEL: (WILDCARD, {}),
                Lexicon.MASK: (WILDCARD, {}),
                Lexicon.FUNC: (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name,
                                                             "tooltip":"Type of adjustment (e.g., blur, sharpen, invert)"}),
                Lexicon.RADIUS: ("INT", {"default": 3, "min": 3, "step": 1}),
                Lexicon.VALUE: ("FLOAT", {"default": 1, "min": 0, "step": 0.1}),
                Lexicon.LOHI: ("VEC2", {"default": (0, 1), "step": 0.01, "precision": 4,
                                        "round": 0.00001, "label": [Lexicon.LO, Lexicon.HI]}),
                Lexicon.LMH: ("VEC3", {"default": (0, 0.5, 1), "step": 0.01, "precision": 4,
                                        "round": 0.00001, "label": [Lexicon.LO, Lexicon.MID, Lexicon.HI]}),
                Lexicon.HSV: ("VEC3",{"default": (0, 1, 1), "step": 0.01, "precision": 4,
                                        "round": 0.00001, "label": [Lexicon.H, Lexicon.S, Lexicon.V]}),
                Lexicon.CONTRAST: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01,
                                                "precision": 4, "round": 0.00001}),
                Lexicon.GAMMA: ("FLOAT", {"default": 1, "min": 0.00001, "max": 1, "step": 0.01,
                                            "precision": 4, "round": 0.00001}),
                Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1,
                                            "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"})
            }
        }
        return Lexicon._parse(d, cls)

    def run(self, **kw)  -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNC, EnumConvertType.STRING, EnumAdjustOP.BLUR.name, enumType=EnumAdjustOP)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3, 3)
        amt = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 0, 0, 1)
        lohi = parse_param(kw, Lexicon.LOHI, EnumConvertType.VEC2, [(0, 1)], 0, 1)
        lmh = parse_param(kw, Lexicon.LMH, EnumConvertType.VEC3, [(0, 0.5, 1)], 0, 1)
        hsv = parse_param(kw, Lexicon.HSV, EnumConvertType.VEC3, [(0, 1, 1)], 0, 1)
        contrast = parse_param(kw, Lexicon.CONTRAST, EnumConvertType.FLOAT, 1, 0, 0)
        gamma = parse_param(kw, Lexicon.GAMMA, EnumConvertType.FLOAT, 1, 0, 1)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, mask, op, radius, amt, lohi,
                                                     lmh, hsv, contrast, gamma, matte, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, radius, amt, lohi, lmh, hsv, contrast, gamma, matte, invert) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            if (cc := channel_count(pA)[0]) == 4:
                alpha = pA[:,:,3]

            match EnumAdjustOP[op]:
                case EnumAdjustOP.INVERT:
                    img_new = image_invert(pA, amt)

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
                    img_new = cv2.GaussianBlur(pA, (r, r), sigmaX=amt)

                case EnumAdjustOP.MEDIAN_BLUR:
                    r = min(radius, 357)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.medianBlur(pA, r)

                case EnumAdjustOP.SHARPEN:
                    r = min(radius, 511)
                    if r % 2 == 0:
                        r += 1
                    img_new = image_sharpen(pA, kernel_size=r, amount=amt)

                case EnumAdjustOP.EMBOSS:
                    img_new = morph_emboss(pA, amt, radius)

                case EnumAdjustOP.EQUALIZE:
                    img_new = image_equalize(pA)

                case EnumAdjustOP.PIXELATE:
                    img_new = image_pixelate(pA, amt / 255.)

                case EnumAdjustOP.QUANTIZE:
                    img_new = image_quantize(pA, int(amt))

                case EnumAdjustOP.POSTERIZE:
                    img_new = image_posterize(pA, int(amt))

                case EnumAdjustOP.OUTLINE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_GRADIENT, (radius, radius))

                case EnumAdjustOP.DILATE:
                    img_new = cv2.dilate(pA, (radius, radius), iterations=int(amt))

                case EnumAdjustOP.ERODE:
                    img_new = cv2.erode(pA, (radius, radius), iterations=int(amt))

                case EnumAdjustOP.OPEN:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_OPEN, (radius, radius), iterations=int(amt))

                case EnumAdjustOP.CLOSE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_CLOSE, (radius, radius), iterations=int(amt))

            h, w, cc = pA.shape
            mask = channel_solid(w, h, 255) if mask is None else tensor2cv(mask)
            mask = image_grayscale(mask)
            if invert:
                mask = 255 - mask
            pA = image_blend(pA, img_new, mask)
            if cc == 4:
                pA[:,:,3] = alpha
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class ColorMatchNode(JOVBaseNode):
    NAME = "COLOR MATCH (JOV) 💞"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    DESCRIPTION = """
The `Color Match` node allows you to adjust the color scheme of one image to match another using various methods. You can choose from different color matching modes such as LUT, Histogram, and Reinhard. Additionally, you can specify options like color maps, the number of colors, and whether to flip or invert the images. This node supports the creation of seamless and cohesive visuals by matching colors accurately, making it ideal for texture work or masking in motion graphics and design projects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {} ,
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.PIXEL_B: (WILDCARD, {}),
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
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1,
                                        "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True}),
        }}
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
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            h, w = pA.shape[:2]
            pB = tensor2cv(pB) if pB is not None else channel_solid(w, h, chan=EnumImageType.BGRA)
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
            matte = pixel_eval(matte, EnumImageType.BGRA)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class ThresholdNode(JOVBaseNode):
    NAME = "THRESHOLD (JOV) 📉"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    DESCRIPTION = """
The `Threshold` node enables you to apply thresholding techniques to images, useful for segmentation and feature extraction. It offers various thresholding modes such as binary and adaptive, along with options to adjust the threshold value and block size. Additionally, you can invert the resulting mask if needed, making it versatile for image processing tasks.
"""
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {} ,
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.ADAPT: ( EnumThresholdAdapt._member_names_,
                            {"default": EnumThresholdAdapt.ADAPT_NONE.name}),
            Lexicon.FUNC: ( EnumThreshold._member_names_, {"default": EnumThreshold.BINARY.name}),
            Lexicon.THRESHOLD: ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.005}),
            Lexicon.SIZE: ("INT", {"default": 3, "min": 3, "max": 103, "step": 1}),
            Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"})
        }}
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
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class ColorBlindNode(JOVBaseNode):
    NAME = "COLOR BLIND (JOV) 👁‍🗨"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    DESCRIPTION = """
The `Color Blind` node facilitates the simulation of color blindness effects on images, aiding in accessibility testing and design adjustments. It offers options to simulate various types of color deficiencies, adjust the severity of the effect, and apply the simulation using different simulators. This node is valuable for ensuring inclusivity in visual content and design processes.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.COLORMATCH_MODE: (EnumCBDefiency._member_names_,
                                        {"default": EnumCBDefiency.PROTAN.name}),
            Lexicon.COLORMATCH_MAP: (EnumCBSimulator._member_names_,
                                        {"default": EnumCBSimulator.AUTOSELECT.name}),
            Lexicon.VALUE: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.001}),
        }}
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        defiency = parse_param(kw, Lexicon.DEFIENCY, EnumConvertType.STRING, EnumCBDefiency.PROTAN.name)
        simulator = parse_param(kw, Lexicon.SIMULATOR, EnumConvertType.STRING, EnumCBSimulator.AUTOSELECT.name)
        severity = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 1)
        params = list(zip_longest_fill(pA, defiency, simulator, severity))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, defiency, simulator, severity) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            defiency = EnumCBDefiency[defiency]
            simulator = EnumCBSimulator[simulator]
            pA = image_color_blind(pA, defiency, simulator, severity)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class FilterMaskNode(JOVBaseNode):
    NAME = "FILTER MASK (JOV) 🤿"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "MASK",)
    SORT = 700
    DESCRIPTION = """
The `Filter Mask` node allows you to create masks based on color ranges within an image, ideal for selective filtering and masking tasks. You can specify the color range using start and end values along with an optional fuzziness factor to adjust the range. This node provides flexibility in defining precise color-based masks for various image processing applications.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
            "required": {},
            "optional": {
                Lexicon.PIXEL_A: (WILDCARD, {}),
                Lexicon.START: ("VEC3", {"default": (128, 128, 128), "step": 1, "rgb": True}),
                Lexicon.BOOLEAN: ("BOOLEAN", {"default": False}),
                Lexicon.END: ("VEC3", {"default": (255, 255, 255), "step": 1, "rgb": True}),
                Lexicon.FLOAT: ("FLOAT", {"default": 0.5, "min":0, "max":1, "step": 0.01, "tooltip": "the fuzziness to add to the start and end range"})
            }
        }
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[Any, ...]:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        start = parse_param(kw, Lexicon.START, EnumConvertType.VEC3, 0, 0, 255)
        toggle_size = parse_param(kw, Lexicon.BOOLEAN, EnumConvertType.VEC3, 0, 0, 255)
        end = parse_param(kw, Lexicon.END, EnumConvertType.VEC3, 0, 0, 1)
        fuzz = parse_param(kw, Lexicon.FLOAT, EnumConvertType.FLOAT, 0, 0, 1)
        toggle_size = parse_param(kw, Lexicon.BOOLEAN, EnumConvertType.VEC3, 0, 0, 255)
        params = list(zip_longest_fill(pA, start, toggle_size, end, fuzz))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, start, toggle_size, end, fuzz) in enumerate(params):
            img = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            start = torch.tensor(start)
            l = (start - fuzz * 128).clamp(min=0).view(1, 1, 1, 3)
            if toggle_size:
                end = torch.tensor(end)
                h = (end + fuzz * 128).clamp(max=255).view(1, 1, 1, 3)
            else:
                h = (start + fuzz * 128).clamp(max=255).view(1, 1, 1, 3)
            mask = (torch.clamp(pA, 0, 1.0) * 255.0).round().to(torch.int)
            mask = ((mask >= l) & (mask <= h)).all(dim=-1)
            alpha = tensor2cv(mask)
            img = cv2.bitwise_and(img, img, mask=alpha)
            images.append([cv2tensor(img), mask.float()])
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]
