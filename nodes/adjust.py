"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Adjustment
"""

from typing import Optional

import cv2
import torch
import numpy as np

from Jovimetrix import tensor2cv, cv2tensor, cv2mask, zip_longest_fill, \
    deep_merge_dict, comfy_to_tuple, \
    JOVImageInOutBaseNode, Lexicon, Logger, \
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
                Lexicon.RADIUS: ("INT", {"default": 1, "min": 3,  "max": 8192, "step": 1}),
                Lexicon.AMT: ("FLOAT", {"default": 1, "min": 0, "step": 0.1}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_INVERT)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        op = kw.get(Lexicon.FUNC,[None])
        radius = kw.get(Lexicon.RADIUS,[None])
        amt = kw.get(Lexicon.AMT,[None])
        invert = kw.get(Lexicon.INVERT,[None])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, op, radius, amt, invert):
            img, o, r, a, i = data

            if img is None:
                zero = torch.zeros((0, 0, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            img = tensor2cv(img)
            o = EnumAdjustOP[o] if o else EnumAdjustOP.BLUR
            r = r if r else 3
            r = r if r % 2 == 1 else r + 1
            a = a if a else 0

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

                case EnumAdjustOP.MEAN:
                    img = comp.color_average(img)

                case EnumAdjustOP.EQUALIZE:
                    img = cv2.equalizeHist(img)

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

            if (i or 0) != 0:
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
                Lexicon.COLORMAP: (['NONE'] + EnumColorMap._member_names_, {"default": EnumColorMap.HSV.name}),
                Lexicon.THRESHOLD: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01},),
                Lexicon.BLUR: ("INT", {"default": 13, "min": 3, "step": 1},),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL2, d, IT_FLIP, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:

        pixelA = kw.get(Lexicon.PIXEL_A, [None])
        pixelB = kw.get(Lexicon.PIXEL_B, [None])
        colormap = kw.get(Lexicon.COLORMAP, [None])
        # if the colormap is not "none" entry...use it.
        # usemap = usemap or [None]
        threshold = kw.get(Lexicon.THRESHOLD, [None])
        blur = kw.get(Lexicon.BLUR, [None])
        flip = kw.get(Lexicon.FLIP, [None])
        invert = kw.get(Lexicon.INVERT, [None])

        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, colormap,
                                     threshold, blur, flip, invert):

            a, b, c, t, bl, f, i = data

            a, b = comp.pixel_convert(a, b)
            if a is None and b is None:
                zero = torch.zeros((0, 0, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue
            a = tensor2cv(a)
            b = tensor2cv(b)

            if f is not None and f:
                a, b = b, a

            if (c := EnumColorMap[c] if c else None) is None:
                a = comp.color_match(a, b)
            else:
                if t is not None and t != 0:
                    bl = bl if bl is not None else 13
                    a = comp.color_match_heat_map(a, t, c, bl)
                else:
                    a = comp.color_match_custom_map(a, colormap=c)

            if (i or 0) != 0:
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
        lohi = comfy_to_tuple(Lexicon.LOHI, kw, [(0, 1)])
        lo, hi = lohi
        invert = kw.get(Lexicon.INVERT, [None])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, lo, hi, invert):
            img, lo, hi, i = data

            if img is None:
                zero = torch.zeros((1, 1, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue
            img = tensor2cv(img)
            lo = lo or 0.27
            hi = hi or 0.72
            img = comp.morph_edge_detect(img, low=lo, high=hi)
            if (i or 0) != 0:
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
        hsv = kw.get(Lexicon.HSV, [None])
        contrast = kw.get(Lexicon.CONTRAST, [None])
        gamma = kw.get(Lexicon.GAMMA, [None])
        invert = kw.get(Lexicon.INVERT, [None])

        masks = []
        images = []
        for data in zip_longest_fill(pixels, hsv, contrast, gamma, invert):

            img, hsv, c, g, i = data
            h, s, v = hsv

            if img is None:
                zero = torch.zeros((0, 0, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue
            img = tensor2cv(img)

            if h != 0 or s != 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                if h != 0:
                    h *= 255
                    img[:, :, 0] = (img[:, :, 0] + h) % 180

                if s != 1:
                    img[:, :, 1] = np.clip(img[:, :, 1] * s, 0, 255)

                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            if c != 0:
                img = comp.light_contrast(img, 1 - c)

            if v != 1:
                img = comp.light_exposure(img, v)

            if g != 1:
                img = comp.light_gamma(img, g)

            if (i or 0) != 0:
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
        hi = kw.get(Lexicon.HI, [None])
        mid = kw.get(Lexicon.MID, [None])
        lo = kw.get(Lexicon.LO, [None])
        gamma = kw.get(Lexicon.GAMMA, [None])
        invert = kw.get(Lexicon.INVERT, [None])

        masks = []
        images = []
        for data in zip_longest_fill(pixels, lo, mid, hi, gamma, invert):
            img, l, m, h, g, i = data

            if img is None:
                zero = torch.zeros((1, 1, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            l = l or 0
            h = h or 1
            img = torch.maximum(img - l, torch.tensor(0.0))
            img = torch.minimum(img, (h - l))
            img = (img + (m or 0.5)) - 0.5
            img = torch.sign(img) * torch.pow(torch.abs(img), 1.0 / (g or 1))
            img = (img + 0.5) / h
            if (i or 0) != 0:
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
        op = kw.get(Lexicon.FUNC, [None])
        adapt = kw.get(Lexicon.ADAPT, [None])
        threshold = kw.get(Lexicon.THRESHOLD, [None])
        size = kw.get(Lexicon.SIZE, [None])
        invert = kw.get(Lexicon.INVERT, [None])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, op, adapt, threshold, size, invert):
            img, o, a, t, b, i = data

            if img is None:
                zero = torch.zeros((1, 1, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue
            img = tensor2cv(img)
            o = EnumThreshold[o] if o else EnumThreshold.BINARY
            a = EnumThresholdAdapt[a] if a else EnumThresholdAdapt.ADAPT_NONE
            t = t if t else 0.5
            b = b if b else 3
            img = comp.adjust_threshold(img, threshold=t, mode=o, adapt=a, block=b, const=t)
            if (i or 0) != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
