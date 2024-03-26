"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Adjustment
"""

from enum import Enum

import cv2
import torch
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOV_WEB_RES_ROOT, JOVBaseNode, WILDCARD
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import EnumConvertType, zip_longest_fill, parse_parameter
from Jovimetrix.sup.image import channel_count, \
    color_match_histogram, color_match_lut, color_match_reinhard, \
    cv2tensor_full, image_color_blind, image_scalefit, tensor2cv, image_equalize, \
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
    NAME_URL = "COMPARISON 🕵🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"[{NAME_URL}]({JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md)"
    DESC = "Blur, Sharpen, Emboss, Levels, HSV, Edge detection."
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.MASK: (WILDCARD, {}),
            Lexicon.FUNC: (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name}),
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
        }}
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        # logger.debug(kw)
        pA = parse_parameter(Lexicon.PIXEL, kw, None, EnumConvertType.IMAGE)
        mask = parse_parameter(Lexicon.MASK, kw, None, EnumConvertType.IMAGE)
        op = parse_parameter(Lexicon.FUNC, kw, EnumAdjustOP.BLUR.name, EnumConvertType.STRING)
        radius = parse_parameter(Lexicon.RADIUS, kw, 3, EnumConvertType.INT, 3)
        amt = parse_parameter(Lexicon.VALUE, kw, 1, EnumConvertType.FLOAT, 0)
        lohi = parse_parameter(Lexicon.LOHI, kw, (0, 1), EnumConvertType.VEC2, 0, 1)
        lmh = parse_parameter(Lexicon.LMH, kw, (0, 0.5, 1), EnumConvertType.VEC3, 0, 1)
        hsv = parse_parameter(Lexicon.HSV, kw, (0, 1, 1), EnumConvertType.VEC3, 0, 1)
        contrast = parse_parameter(Lexicon.CONTRAST, kw, 0, EnumConvertType.FLOAT, 0, 1)
        gamma = parse_parameter(Lexicon.GAMMA, kw, 1, EnumConvertType.FLOAT, 0, 1)
        matte = parse_parameter(Lexicon.MATTE, kw, (0, 0, 0, 255), EnumConvertType.VEC4INT, 0, 255)
        invert = parse_parameter(Lexicon.INVERT, kw, False, EnumConvertType.BOOLEAN)
        params = [tuple(x) for x in zip_longest_fill(pA, mask, op, radius, amt, lohi,
                                                     lmh, hsv, contrast, gamma, matte, invert)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, radius, amt, lohi, lmh, hsv, contrast, gamma, matte, invert) in enumerate(params):
            # logger.debug(radius)
            pA = tensor2cv(pA)
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
                    img_new = cv2.dilate(pA, (r, r), iterations=int(amt))

                case EnumAdjustOP.ERODE:
                    img_new = cv2.erode(pA, (r, r), iterations=int(amt))

                case EnumAdjustOP.OPEN:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_OPEN, (radius, radius), iterations=int(amt))

                case EnumAdjustOP.CLOSE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_CLOSE, (radius, radius), iterations=int(amt))

            mask = tensor2cv(mask, chan=EnumImageType.GRAYSCALE)
            if not invert:
                mask = 255 - mask

            if (wh := pA.shape[:2]) != mask.shape[:2]:
                mask = cv2.resize(mask, wh[::-1])
            pA = image_blend(pA, img_new, mask)
            if cc == 4:
                pA[:,:,3] = alpha
            matte = pixel_eval(matte, EnumImageType.BGRA)
            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]

class ColorMatchNode(JOVBaseNode):
    NAME = "COLOR MATCH (JOV) 💞"
    NAME_URL = "COMPARISON 🕵🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"[{NAME_URL}]({JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md)"

    DESC = "Project the colors of one image  onto another or use a pre-defined color target."
        RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)

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
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_parameter(Lexicon.PIXEL_A, kw, None, EnumConvertType.IMAGE)
        pB = parse_parameter(Lexicon.PIXEL_B, kw, None, EnumConvertType.IMAGE)
        colormatch_mode = parse_parameter(Lexicon.COLORMATCH_MODE, kw, EnumColorMatchMode.REINHARD.name, EnumConvertType.STRING)
        colormatch_map = parse_parameter(Lexicon.COLORMATCH_MAP, kw, EnumColorMatchMap.USER_MAP.name, EnumConvertType.STRING)
        colormap = parse_parameter(Lexicon.COLORMAP, kw, EnumColorMap.HSV.name, EnumConvertType.STRING)
        num_colors = parse_parameter(Lexicon.VALUE, kw, 255, EnumConvertType.INT)
        flip = parse_parameter(Lexicon.FLIP, kw, False, EnumConvertType.BOOLEAN)
        invert = parse_parameter(Lexicon.INVERT, kw, False, EnumConvertType.BOOLEAN)
        matte = parse_parameter(Lexicon.MATTE, kw, (0, 0, 0, 255), EnumConvertType.VEC4INT, 0, 255)
        params = [tuple(x) for x in zip_longest_fill(pA, pB, colormap, colormatch_mode,
                                                     colormatch_map, num_colors, flip, invert, matte)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, colormap, mode, cmap, num_colors, flip, invert, matte) in enumerate(params):
            if flip == True:
                pA, pB = pB, pA
            pA = tensor2cv(pA)
            h, w = pA.shape[:2]
            pB = tensor2cv(pB, width=w, height=h)
            mode = EnumColorMatchMode[mode]
            match mode:
                case EnumColorMatchMode.LUT:
                    cmap = EnumColorMatchMap[cmap]
                    if cmap == EnumColorMatchMap.PRESET_MAP:
                        pB = None
                    colormap = EnumColorMap[colormap]
                    pA = color_match_lut(pA, colormap.value, pB, num_colors)
                case EnumColorMatchMode.HISTOGRAM:
                    pB = image_scalefit(pB, w, h, EnumScaleMode.CROP, (0,0,0,0))
                    pB = image_scalefit(pB, w, h, EnumScaleMode.MATTE, (0,0,0,0))
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
    NAME_URL = "COMPARISON 🕵🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"[{NAME_URL}]({JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md)"

    DESC = "Clip an input based on a mid point value."
        RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)

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
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_parameter(Lexicon.PIXEL, kw, None, EnumConvertType.IMAGE)
        mode = parse_parameter(Lexicon.FUNC, kw, EnumThreshold.BINARY.name, EnumConvertType.STRING)
        adapt = parse_parameter(Lexicon.ADAPT, kw, EnumThresholdAdapt.ADAPT_NONE.name, EnumConvertType.STRING)
        threshold = parse_parameter(Lexicon.THRESHOLD, kw, 1, EnumConvertType.FLOAT, 0, 1)
        block = parse_parameter(Lexicon.SIZE, kw, 3, EnumConvertType.INT, 3)
        invert = parse_parameter(Lexicon.INVERT, kw, False, EnumConvertType.BOOLEAN)
        params = [tuple(x) for x in zip_longest_fill(pA, mode, adapt, threshold, block, invert)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mode, adapt, th, block, invert) in enumerate(params):
            pA = tensor2cv(pA)
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
    NAME_URL = "COMPARISON 🕵🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = f"[{NAME_URL}]({JOV_WEB_RES_ROOT}/node/{NAME_URL}/{NAME_URL}.md)"
    HELP_URL = f"{JOV_CATEGORY}#-{NAME_URL}"
    DESC = "Transform an image into specific color blind color space"
        RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)

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
        return Lexicon._parse(d, cls.HELP_URL)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = parse_parameter(Lexicon.PIXEL, kw, None, EnumConvertType.IMAGE)
        defiency = parse_parameter(Lexicon.DEFIENCY, kw, EnumCBDefiency.PROTAN.name, EnumConvertType.STRING)
        simulator = parse_parameter(Lexicon.SIMULATOR, kw, EnumCBSimulator.AUTOSELECT.name, EnumConvertType.STRING)
        severity = parse_parameter(Lexicon.VALUE, kw, 1, EnumConvertType.FLOAT)
        params = [tuple(x) for x in zip_longest_fill(pA, defiency, simulator, severity)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, defiency, simulator, severity) in enumerate(params):
            pA = tensor2cv(pA)
            defiency = EnumCBDefiency[defiency]
            simulator = EnumCBSimulator[simulator]
            pA = image_color_blind(pA, defiency, simulator, severity)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return [torch.stack(i, dim=0).squeeze(1) for i in list(zip(*images))]
