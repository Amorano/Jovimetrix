"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Adjustment
"""

import cv2
import torch

from Jovimetrix import MIN_IMAGE_SIZE, tensor2cv, cv2tensor, cv2mask, zip_longest_fill, \
    deep_merge_dict, parse_tuple, parse_number, \
    JOVImageInOutBaseNode, Lexicon, EnumTupleType, \
    IT_PIXELS, IT_PIXEL2, IT_HSV, IT_FLIP, IT_LOHI, IT_LMH, IT_INVERT, IT_CONTRAST, \
    IT_GAMMA, IT_REQUIRED

from Jovimetrix.sup import comp
from Jovimetrix.sup.comp import EnumAdjustOP, EnumThresholdAdapt, EnumColorMap, EnumThreshold

# =============================================================================

class AdjustNode(JOVImageInOutBaseNode):
    NAME = "ADJUST (JOV) 🕸️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Blur, Sharpen and Emboss an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.FUNC: (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name}),
                Lexicon.RADIUS: ("INT", {"default": 1, "min": 3, "step": 1}),
                Lexicon.AMT: ("FLOAT", {"default": 1, "min": 0, "step": 0.1}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_INVERT)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        op = kw.get(Lexicon.FUNC,[EnumAdjustOP.BLUR])
        radius = kw.get(Lexicon.RADIUS,[3])
        amt = kw.get(Lexicon.AMT,[1])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        masks = []
        images = []
        for data in zip_longest_fill(pixels, op, radius, amt, i):
            img, o, r, a, i = data

            if img is None:
                zero = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            img = tensor2cv(img)
            o = EnumAdjustOP[o]

            match o:
                case EnumAdjustOP.BLUR:
                    img = cv2.blur(img, (r, r))

                case EnumAdjustOP.STACK_BLUR:
                    r = min(r, 1399)
                    img = cv2.stackBlur(img, (r, r))

                case EnumAdjustOP.GAUSSIAN_BLUR:
                    r = min(r, 999)
                    img = cv2.GaussianBlur(img, (r, r), sigmaX=float(a))

                case EnumAdjustOP.MEDIAN_BLUR:
                    r = min(r, 357)
                    img = cv2.medianBlur(img, r)

                case EnumAdjustOP.SHARPEN:
                    r = min(r, 511)
                    img = comp.adjust_sharpen(img, kernel_size=r, amount=a)

                case EnumAdjustOP.EMBOSS:
                    img = comp.morph_emboss(img, a, r)

                #case EnumAdjustOP.MEAN:
                #    img = comp.color_average(img)

                case EnumAdjustOP.EQUALIZE:
                    img = comp.adjust_equalize(img)

                case EnumAdjustOP.PIXELATE:
                    img = comp.adjust_pixelate(img, a / 255.)

                case EnumAdjustOP.QUANTIZE:
                    img = comp.adjust_quantize(img, int(a))

                case EnumAdjustOP.POSTERIZE:
                    img = comp.adjust_posterize(img, int(a))

                case EnumAdjustOP.OUTLINE:
                    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, (r, r))

                case EnumAdjustOP.DILATE:
                    img = cv2.dilate(img, (r, r), iterations=int(a))

                case EnumAdjustOP.ERODE:
                    img = cv2.erode(img, (r, r), iterations=int(a))

                case EnumAdjustOP.OPEN:
                    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (r, r), iterations=int(a))

                case EnumAdjustOP.CLOSE:
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (r, r), iterations=int(a))

            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class ColorMatchNode(JOVImageInOutBaseNode):
    NAME = "COLOR MATCH (JOV) 💞"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Project the colors of one pixel block onto another"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.COLORMAP: (['NONE'] + EnumColorMap._member_names_, {"default": 'NONE'}),
                Lexicon.THRESHOLD: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01},),
                Lexicon.BLUR: ("INT", {"default": 13, "min": 3, "step": 1},),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL2, d, IT_FLIP, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixelA = kw.get(Lexicon.PIXEL_A, [None])
        pixelB = kw.get(Lexicon.PIXEL_B, [None])
        colormap = kw.get(Lexicon.COLORMAP, ['NONE'])
        # if the colormap is not "none" entry...use it.
        # usemap = usemap or [None]
        threshold = parse_number(Lexicon.THRESHOLD, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        blur = kw.get(Lexicon.BLUR, [3])
        flip = kw.get(Lexicon.FLIP, [False])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, colormap, threshold, blur, flip, i):
            a, b, c, t, bl, f, i = data
            a, b = comp.pixel_convert(a, b)
            if a is None and b is None:
                zero = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            a = tensor2cv(a)
            b = tensor2cv(b)
            if f is not None and f:
                a, b = b, a

            if c == 'NONE':
                a = comp.color_match(a, b)
            else:
                c = EnumColorMap[c].value
                if t != 0:
                    bl = bl if bl is not None else 13
                    a = comp.color_match_heat_map(a, t, c, bl)
                else:
                    a = comp.color_match_custom_map(a, colormap=c)

            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class FindEdgeNode(JOVImageInOutBaseNode):
    NAME = "FIND EDGES (JOV) 🔳"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Find Edges on an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_LOHI, IT_INVERT)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        lohi = parse_tuple(Lexicon.LOHI, kw, EnumTupleType.FLOAT, (0, 1), clip_min=0, clip_max=1)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        masks = []
        images = []
        for img, lohi, i in zip_longest_fill(pixels, lohi, i):
            lo, hi = lohi
            if img is None:
                zero = torch.zeros((1, 1, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            img = tensor2cv(img)
            img = comp.morph_edge_detect(img, low=lo, high=hi)

            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class HSVNode(JOVImageInOutBaseNode):
    NAME = "HSV (JOV) 🌈"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Adjust the Hue, Saturation, Value, Contrast and Gamma of the input."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_HSV, IT_CONTRAST, IT_GAMMA, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        hsv = parse_tuple(Lexicon.HSV, kw, EnumTupleType.FLOAT, (0, 1, 1), clip_min=0, clip_max=1)
        contrast = parse_number(Lexicon.CONTRAST, kw, EnumTupleType.FLOAT, [0], clip_min=0, clip_max=1)
        gamma = parse_number(Lexicon.GAMMA, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [0], clip_min=0, clip_max=1)
        masks = []
        images = []
        for data in zip_longest_fill(pixels, hsv, contrast, gamma, i):
            img, hsv, c, g, i = data
            h, s, v = hsv
            if img is None:
                zero = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            img = tensor2cv(img)
            img = comp.light_hsv(img, h, s, v)
            if c != 0:
                img = comp.light_contrast(img, 1 - c)

            if g != 0:
                img = comp.light_gamma(img, g)

            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class LevelsNode(JOVImageInOutBaseNode):
    NAME = "LEVELS (JOV) 🛗"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Clip an input based on a low, high and mid point value"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_LMH, IT_GAMMA, IT_INVERT)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        lmh = parse_tuple(Lexicon.LMH, kw, EnumTupleType.FLOAT, (0, 0.5, 1), clip_min=0, clip_max=1)
        gamma = parse_number(Lexicon.GAMMA, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        masks = []
        images = []
        for data in zip_longest_fill(pixels, lmh, gamma, i):
            img, l, m, h, g, i = data

            if img is None:
                zero = torch.zeros((1, 1, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            img = torch.maximum(img - l, torch.tensor(0.0))
            img = torch.minimum(img, (h - l))
            img = (img + (m or 0.5)) - 0.5
            img = torch.sign(img) * torch.pow(torch.abs(img), 1.0 / (g or 1))
            img = (img + 0.5) / h

            if i != 0:
                img = 1 - i - img

            images.append(img)
            masks.append(img)

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class ThresholdNode(JOVImageInOutBaseNode):
    NAME = "THRESHOLD (JOV) 📉"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Clip an input to explicit 0 or 1"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.ADAPT: ( EnumThresholdAdapt._member_names_, {"default": EnumThresholdAdapt.ADAPT_NONE.name}),
                Lexicon.FUNC: ( EnumThreshold._member_names_, {"default": EnumThreshold.BINARY.name}),
                Lexicon.THRESHOLD: ("FLOAT", {"default": 0.5, "min": -100, "max": 100, "step": 0.01},),
                Lexicon.SIZE: ("INT", {"default": 3, "min": 3, "max": 103, "step": 1},),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_INVERT)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:

        pixels = kw.get(Lexicon.PIXEL, [None])
        op = kw.get(Lexicon.FUNC, [EnumThreshold.BINARY])
        adapt = kw.get(Lexicon.ADAPT, [EnumThresholdAdapt.ADAPT_NONE])
        threshold = kw.get(Lexicon.THRESHOLD, [0.5])
        size = kw.get(Lexicon.SIZE, [3])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        masks = []
        images = []
        for data in zip_longest_fill(pixels, op, adapt, threshold, size, i):
            img, o, a, t, b, i = data
            if img is None:
                zero = torch.zeros((1, 1, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            img = tensor2cv(img)
            o = EnumThreshold[o]
            a = EnumThresholdAdapt[a]
            img = comp.adjust_threshold(img, threshold=t, mode=o, adapt=a, block=b, const=t)
            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )
