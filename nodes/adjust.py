"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Adjustment
"""

import cv2
import torch
from loguru import logger

import comfy

from Jovimetrix import IT_MATTE, JOV_HELP_URL, JOVImageMultiple, \
    IT_PIXEL, IT_PIXEL2, IT_PIXEL_MASK, IT_HSV, IT_FLIP, \
    IT_LOHI, IT_LMH, IT_INVERT, IT_CONTRAST, IT_GAMMA, IT_REQUIRED

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import zip_longest_fill, deep_merge_dict, \
    parse_tuple, \
    parse_number, EnumTupleType

from Jovimetrix.sup.image import EnumImageType, cv2tensor_full, \
    image_equalize, image_levels, image_matte, \
    image_posterize, image_pixelate, pixel_eval, tensor2cv, \
    image_quantize, image_sharpen, image_threshold,  \
    image_blend, image_invert, morph_edge_detect, \
    morph_emboss, image_contrast, image_hsv, image_gamma, color_match, \
    color_match_custom_map, color_match_heat_map, \
    EnumColorMap, EnumAdjustOP, EnumThresholdAdapt, EnumThreshold

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"

# =============================================================================

class AdjustNode(JOVImageMultiple):
    NAME = "ADJUST (JOV) 🕸️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Blur, Sharpen, Emboss, Levels, HSV, Edge detection."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.FUNC: (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name}),
                Lexicon.RADIUS: ("INT", {"default": 3, "min": 3, "step": 1}),
                Lexicon.VALUE: ("FLOAT", {"default": 1, "min": 0, "step": 0.1}),
            }}
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL_MASK, d, IT_LOHI, IT_LMH, IT_HSV,
                            IT_CONTRAST, IT_GAMMA, IT_MATTE, IT_INVERT)
        return Lexicon._parse(d, JOV_HELP_URL + "/ADJUST#-adjust")

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        img = kw.get(Lexicon.PIXEL, [None])
        mask = kw.get(Lexicon.MASK, [None])
        op = kw.get(Lexicon.FUNC, [EnumAdjustOP.BLUR])
        radius = kw.get(Lexicon.RADIUS, [3])
        amt = kw.get(Lexicon.VALUE, [0])
        lohi = parse_tuple(Lexicon.LOHI, kw, EnumTupleType.FLOAT, (0, 1), clip_min=0, clip_max=1)
        lmh = parse_tuple(Lexicon.LMH, kw, EnumTupleType.FLOAT, (0, 0.5, 1), clip_min=0, clip_max=1)
        hsv = parse_tuple(Lexicon.HSV, kw, EnumTupleType.FLOAT, (0, 1, 1), clip_min=0, clip_max=1)
        contrast = parse_number(Lexicon.CONTRAST, kw, EnumTupleType.FLOAT, [0], clip_min=0, clip_max=1)
        gamma = parse_number(Lexicon.GAMMA, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0), clip_min=0)
        invert = kw.get(Lexicon.INVERT, [False])
        params = [tuple(x) for x in zip_longest_fill(img, mask, op, radius, amt, lohi,
                                                     lmh, hsv, contrast, gamma, matte, invert)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pixel, mask, o, r, a, lohi, lmh, hsv, con, gamma, matte, invert) in enumerate(params):
            img = tensor2cv(pixel)
            match EnumAdjustOP[o]:
                case EnumAdjustOP.INVERT:
                    img_new = image_invert(img, a)

                case EnumAdjustOP.LEVELS:
                    l, m, h = lmh
                    img_new = image_levels(img, l, h, m, gamma)

                case EnumAdjustOP.HSV:
                    h, s, v = hsv
                    img_new = image_hsv(img, h, s, v)
                    if con != 0:
                        img_new = image_contrast(img_new, 1 - con)

                    if gamma != 0:
                        img_new = image_gamma(img_new, gamma)

                case EnumAdjustOP.FIND_EDGES:
                    lo, hi = lohi
                    img_new = morph_edge_detect(img, low=lo, high=hi)

                case EnumAdjustOP.BLUR:
                    img_new = cv2.blur(img, (r, r))

                case EnumAdjustOP.STACK_BLUR:
                    r = min(r, 1399)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.stackBlur(img, (r, r))

                case EnumAdjustOP.GAUSSIAN_BLUR:
                    r = min(r, 999)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.GaussianBlur(img, (r, r), sigmaX=float(a))

                case EnumAdjustOP.MEDIAN_BLUR:
                    r = min(r, 357)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.medianBlur(img, r)

                case EnumAdjustOP.SHARPEN:
                    r = min(r, 511)
                    if r % 2 == 0:
                        r += 1
                    img_new = image_sharpen(img, kernel_size=r, amount=a)

                case EnumAdjustOP.EMBOSS:
                    img_new = morph_emboss(img, a, r)

                case EnumAdjustOP.EQUALIZE:
                    img_new = image_equalize(img)

                case EnumAdjustOP.PIXELATE:
                    img_new = image_pixelate(img, a / 255.)

                case EnumAdjustOP.QUANTIZE:
                    img_new = image_quantize(img, int(a))

                case EnumAdjustOP.POSTERIZE:
                    img_new = image_posterize(img, int(a))

                case EnumAdjustOP.OUTLINE:
                    img_new = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, (r, r))

                case EnumAdjustOP.DILATE:
                    img_new = cv2.dilate(img, (r, r), iterations=int(a))

                case EnumAdjustOP.ERODE:
                    img_new = cv2.erode(img, (r, r), iterations=int(a))

                case EnumAdjustOP.OPEN:
                    img_new = cv2.morphologyEx(img, cv2.MORPH_OPEN, (r, r), iterations=int(a))

                case EnumAdjustOP.CLOSE:
                    img_new = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (r, r), iterations=int(a))

            mask = tensor2cv(mask, chan=EnumImageType.GRAYSCALE)
            if invert:
                mask = 255 - mask
            if (wh := img.shape[:2]) != mask.shape[:2]:
                mask = cv2.resize(mask, wh[::-1])
            img = image_blend(img, img_new, mask)
            img = cv2tensor_full(img, matte)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))

class ColorMatchNode(JOVImageMultiple):
    NAME = "COLOR MATCH (JOV) 💞"
    CATEGORY = CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Project the colors of one image  onto another or use a pre-defined color target."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.COLORMAP: (['NONE'] + EnumColorMap._member_names_, {"default": 'NONE'}),
                Lexicon.THRESHOLD: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01},),
                Lexicon.BLUR: ("INT", {"default": 13, "min": 3, "step": 1},),
            }}
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL2, d, IT_FLIP, IT_INVERT)
        return Lexicon._parse(d, JOV_HELP_URL + "/ADJUST#-colormatch")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixelA = kw.get(Lexicon.PIXEL_A, [None])
        pixelB = kw.get(Lexicon.PIXEL_B, [None])
        colormap = kw.get(Lexicon.COLORMAP, [EnumColorMap.HSV])
        threshold = parse_number(Lexicon.THRESHOLD, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        blur = kw.get(Lexicon.BLUR, [3])
        flip = kw.get(Lexicon.FLIP, [False])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        params = [tuple(x) for x in zip_longest_fill(pixelA, pixelB, colormap, threshold, blur, flip, i)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (a, b, c, t, bl, f, i) in enumerate(params):
            img = tensor2cv(a)
            b = tensor2cv(b)
            if f is not None and f:
                img, b = b, img
            if c == 'NONE':
                img = color_match(img, b)
            else:
                c = EnumColorMap[c].value
                if t != 0:
                    img = color_match_heat_map(img, t, c, bl)
                else:
                    img = color_match_custom_map(img, colormap=c)

            if i != 0:
                img = image_invert(img, i)

            img = cv2tensor_full(img)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))

class ThresholdNode(JOVImageMultiple):
    NAME = "THRESHOLD (JOV) 📉"
    CATEGORY = CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Clip an input based on a mid point value."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.ADAPT: ( EnumThresholdAdapt._member_names_, {"default": EnumThresholdAdapt.ADAPT_NONE.name}),
                Lexicon.FUNC: ( EnumThreshold._member_names_, {"default": EnumThreshold.BINARY.name}),
                Lexicon.THRESHOLD: ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.005},),
                Lexicon.SIZE: ("INT", {"default": 3, "min": 3, "max": 103, "step": 1},),
            }}
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL, d, IT_INVERT)
        return Lexicon._parse(d, JOV_HELP_URL + "/ADJUST#-threshold")

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        img = kw.get(Lexicon.PIXEL, [None])
        op = kw.get(Lexicon.FUNC, [EnumThreshold.BINARY])
        adapt = kw.get(Lexicon.ADAPT, [EnumThresholdAdapt.ADAPT_NONE])
        threshold = parse_number(Lexicon.THRESHOLD, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        size = kw.get(Lexicon.SIZE, [3])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        params = [tuple(x) for x in zip_longest_fill(img, op, adapt, threshold, size, i)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, o, a, t, b, i) in enumerate(params):
            img = tensor2cv(img)
            o = EnumThreshold[o]
            a = EnumThresholdAdapt[a]
            img = image_threshold(img, threshold=t, mode=o, adapt=a, block=b, const=t)
            if i != 0:
                img = image_invert(img, i)
            logger.debug(img.shape)
            img = cv2tensor_full(img)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))
