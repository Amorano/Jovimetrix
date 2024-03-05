"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Adjustment
"""

from enum import Enum

import cv2
import torch
from loguru import logger

import comfy

from Jovimetrix import JOV_HELP_URL, MIN_IMAGE_SIZE, WILDCARD, JOVImageMultiple
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import zip_longest_fill, parse_tuple, parse_number, EnumTupleType
from Jovimetrix.sup.image import EnumCBDefiency, EnumCBSimulator, EnumScaleMode, batch_extract, channel_count, \
    channel_solid, color_match_histogram, color_match_lut, color_match_reinhard, \
    cv2tensor_full, image_color_blind, image_scalefit, tensor2cv, image_equalize, image_levels, pixel_eval, \
    image_posterize, image_pixelate, image_quantize, image_sharpen, \
    image_threshold, image_blend, image_invert, morph_edge_detect, \
    morph_emboss, image_contrast, image_hsv, image_gamma, \
    EnumImageType, EnumColorMap, EnumAdjustOP, EnumThresholdAdapt, EnumThreshold

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"

class EnumColorMatchMode(Enum):
    REINHARD = 30
    LUT = 10
    HISTOGRAM = 20

class EnumColorMatchMap(Enum):
    USER_MAP = 0
    PRESET_MAP = 10

# =============================================================================

class AdjustNode(JOVImageMultiple):
    NAME = "ADJUST (JOV) 🕸️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Blur, Sharpen, Emboss, Levels, HSV, Edge detection."

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
        return Lexicon._parse(d, JOV_HELP_URL + "/ADJUST#-adjust")

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
        mask = kw.get(Lexicon.MASK, None)
        mask = [None] if mask is None else batch_extract(mask)
        op = kw[Lexicon.FUNC]
        radius = kw[Lexicon.RADIUS]
        amt = kw[Lexicon.VALUE]
        lohi = parse_tuple(Lexicon.LOHI, kw, EnumTupleType.FLOAT, (0, 1), clip_min=0, clip_max=1)
        lmh = parse_tuple(Lexicon.LMH, kw, EnumTupleType.FLOAT, (0, 0.5, 1), clip_min=0, clip_max=1)
        hsv = parse_tuple(Lexicon.HSV, kw, EnumTupleType.FLOAT, (0, 1, 1), clip_min=0, clip_max=1)
        contrast = parse_number(Lexicon.CONTRAST, kw, EnumTupleType.FLOAT, [0], clip_min=0, clip_max=1)
        gamma = parse_number(Lexicon.GAMMA, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)
        invert = kw[Lexicon.INVERT]
        params = [tuple(x) for x in zip_longest_fill(pA, mask, op, radius, amt, lohi,
                                                     lmh, hsv, contrast, gamma, matte, invert)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pA, mask, o, r, a, lohi, lmh, hsv, con, gamma, matte, invert) in enumerate(params):
            if pA is not None:
                pA = tensor2cv(pA)
            else:
                pA = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, chan=EnumImageType.BGRA)
            if (cc := channel_count(pA)[0]) == 4:
                alpha = pA[:,:,3]

            match EnumAdjustOP[o]:
                case EnumAdjustOP.INVERT:
                    img_new = image_invert(pA, a)

                case EnumAdjustOP.LEVELS:
                    l, m, h = lmh
                    img_new = image_levels(pA, l, h, m, gamma)

                case EnumAdjustOP.HSV:
                    h, s, v = hsv
                    img_new = image_hsv(pA, h, s, v)
                    if con != 0:
                        img_new = image_contrast(img_new, 1 - con)

                    if gamma != 0:
                        img_new = image_gamma(img_new, gamma)

                case EnumAdjustOP.FIND_EDGES:
                    lo, hi = lohi
                    img_new = morph_edge_detect(pA, low=lo, high=hi)

                case EnumAdjustOP.BLUR:
                    img_new = cv2.blur(pA, (r, r))

                case EnumAdjustOP.STACK_BLUR:
                    r = min(r, 1399)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.stackBlur(pA, (r, r))

                case EnumAdjustOP.GAUSSIAN_BLUR:
                    r = min(r, 999)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.GaussianBlur(pA, (r, r), sigmaX=float(a))

                case EnumAdjustOP.MEDIAN_BLUR:
                    r = min(r, 357)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.medianBlur(pA, r)

                case EnumAdjustOP.SHARPEN:
                    r = min(r, 511)
                    if r % 2 == 0:
                        r += 1
                    img_new = image_sharpen(pA, kernel_size=r, amount=a)

                case EnumAdjustOP.EMBOSS:
                    img_new = morph_emboss(pA, a, r)

                case EnumAdjustOP.EQUALIZE:
                    img_new = image_equalize(pA)

                case EnumAdjustOP.PIXELATE:
                    img_new = image_pixelate(pA, a / 255.)

                case EnumAdjustOP.QUANTIZE:
                    img_new = image_quantize(pA, int(a))

                case EnumAdjustOP.POSTERIZE:
                    img_new = image_posterize(pA, int(a))

                case EnumAdjustOP.OUTLINE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_GRADIENT, (r, r))

                case EnumAdjustOP.DILATE:
                    img_new = cv2.dilate(pA, (r, r), iterations=int(a))

                case EnumAdjustOP.ERODE:
                    img_new = cv2.erode(pA, (r, r), iterations=int(a))

                case EnumAdjustOP.OPEN:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_OPEN, (r, r), iterations=int(a))

                case EnumAdjustOP.CLOSE:
                    img_new = cv2.morphologyEx(pA, cv2.MORPH_CLOSE, (r, r), iterations=int(a))

            if mask is not None:
                mask = tensor2cv(mask, chan=EnumImageType.GRAYSCALE)
            else:
                mask = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, chan=EnumImageType.GRAYSCALE)
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
        return list(zip(*images))

class ColorMatchNode(JOVImageMultiple):
    NAME = "COLOR MATCH (JOV) 💞"
    CATEGORY = CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Project the colors of one image  onto another or use a pre-defined color target."

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
        return Lexicon._parse(d, JOV_HELP_URL + "/ADJUST#-color-match")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL_A, None)
        pA = [None] if pA is None else batch_extract(pA)
        pB = kw.get(Lexicon.PIXEL_B, None)
        pB = [None] if pB is None else batch_extract(pB)
        colormatch_mode = kw[Lexicon.COLORMATCH_MODE]
        colormatch_map = kw[Lexicon.COLORMATCH_MAP]
        colormap = kw[Lexicon.COLORMAP]
        num_colors = kw[Lexicon.VALUE]
        flip = kw[Lexicon.FLIP]
        invert = kw[Lexicon.INVERT]
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)
        params = [tuple(x) for x in zip_longest_fill(pA, pB, colormap, colormatch_mode,
                                                     colormatch_map, num_colors, flip, invert, matte)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pA, pB, colormap, mode, cmap, num_colors, flip, invert, matte) in enumerate(params):
            if flip == True:
                pA, pB = pB, pA
            if pA is None:
                pA = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, chan=EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)
            h, w = pA.shape[:2]
            if pB is None:
                pB = channel_solid(w, h, chan=EnumImageType.BGRA)
            else:
                pB = tensor2cv(pB)
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
        return list(zip(*images))

class ThresholdNode(JOVImageMultiple):
    NAME = "THRESHOLD (JOV) 📉"
    CATEGORY = CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Clip an input based on a mid point value."

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
        return Lexicon._parse(d, JOV_HELP_URL + "/ADJUST#-threshold")

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
        mode = kw[Lexicon.FUNC]
        adapt = kw[Lexicon.ADAPT]
        threshold = parse_number(Lexicon.THRESHOLD, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        block = kw[Lexicon.SIZE]
        invert = kw[Lexicon.INVERT]
        params = [tuple(x) for x in zip_longest_fill(pA, mode, adapt, threshold, block, invert)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pA, mode, adapt, th, block, invert) in enumerate(params):
            if pA is None:
                pA = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE)
            else:
                pA = tensor2cv(pA)
            mode = EnumThreshold[mode]
            adapt = EnumThresholdAdapt[adapt]
            pA = image_threshold(pA, th, mode, adapt, block)
            if invert == True:
                pA = image_invert(pA, 1)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return list(zip(*images))

class ColorBlindNode(JOVImageMultiple):
    NAME = "COLOR BLIND (JOV) 👁‍🗨"
    CATEGORY = CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Transform an image into specific color blind color space"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {}),
            Lexicon.COLORMATCH_MODE: (EnumCBDefiency._member_names_,
                                        {"default": EnumCBDefiency.PROTAN.name}),
            Lexicon.COLORMATCH_MAP: (EnumCBSimulator._member_names_,
                                        {"default": EnumCBSimulator.AUTOSELECT.name}),
            Lexicon.VALUE: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.001}),
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/ADJUST#-color-match")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL_A, None)
        pA = [None] if pA is None else batch_extract(pA)
        defiency = kw[Lexicon.DEFIENCY]
        simulator = kw[Lexicon.SIMULATOR]
        severity = kw[Lexicon.SIMULATOR]
        params = [tuple(x) for x in zip_longest_fill(pA, defiency, simulator, severity)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pA, defiency, simulator, severity) in enumerate(params):
            if pA is None:
                pA = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, chan=EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)
            pA = image_color_blind(pA, defiency, simulator, severity)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return list(zip(*images))

