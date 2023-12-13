"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Adjustment
"""

from typing import Optional

import cv2
import torch
import numpy as np

from Jovimetrix import Logger, tensor2cv, cv2tensor, cv2mask, \
    zip_longest_fill, deep_merge_dict, \
    JOVImageInOutBaseNode, \
    IT_PIXELS, IT_PIXEL2, IT_INVERT, IT_REQUIRED, IT_PIXELS_REQUIRED

from Jovimetrix.sup import comp
from Jovimetrix.sup.comp import EnumAdjustOP, EnumThresholdAdapt, EnumColorMap, EnumThreshold

# =============================================================================

class AdjustNode(JOVImageInOutBaseNode):
    NAME = "🕸️ Adjust (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Blur, Sharpen and Emboss an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "func": (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name}),
                "radius": ("INT", {"default": 1, "min": 3,  "max": 8192, "step": 1}),
                "amount": ("FLOAT", {"default": 1, "min": 0, "step": 0.1}),
            }}
        return deep_merge_dict(IT_PIXELS_REQUIRED, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            func: Optional[list[str]]=None,
            radius: Optional[list[float]]=None,
            amount: Optional[list[float]]=None,
            invert: Optional[list[float]]=None)  -> tuple[torch.Tensor, torch.Tensor]:

        func = func or [None]
        radius = radius or [None]
        amount = amount or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, func, radius, amount, invert):
            img, o, r, a, i = data
            img = tensor2cv(img)

            o = (EnumAdjustOP[o] if o is not None else EnumAdjustOP.BLUR)
            r = r if r is not None else 3
            r = r if r % 2 == 1 else r + 1

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

            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class ColorMatchNode(JOVImageInOutBaseNode):
    NAME = "💞 Color Match (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Project the colors of one pixel block onto another"

    @classmethod
    def INPUT_TYPES(s) -> dict:
        d = {"optional": {
                "colormap": (EnumColorMap._member_names_, {"default": EnumColorMap.HSV.name}),
                "usemap": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01},),
                "blur": ("INT", {"default": 13, "min": 3, "step": 1},),
                "flip": ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL2, d, IT_INVERT)

    def run(self,
            pixelA: Optional[list[torch.tensor]]=None,
            pixelB: Optional[list[torch.tensor]]=None,
            colormap: Optional[list[str]]=None,
            usemap: Optional[list[bool]]=None,
            threshold: Optional[list[float]]=None,
            blur: Optional[list[float]]=None,
            flip: Optional[list[bool]]=None,
            invert: Optional[list[float]]=None) -> tuple[torch.Tensor, torch.Tensor]:

        pixelA = pixelA or [None]
        pixelB = pixelB or [None]
        colormap = colormap or [None]
        usemap = usemap or [None]
        threshold = threshold or [None]
        blur = blur or [None]
        flip = flip or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, colormap,
                                     usemap, threshold, blur, flip, invert):

            a, b, c, u, t, bl, f, i = data
            a = tensor2cv(a)
            if b is not None:
                b = tensor2cv(b)

            if f is not None and f:
                a, b = b, a

            if (u is not None and u):
                c = EnumColorMap[c].value if c is not None else EnumColorMap.HSV
                if t is not None and t != 0:
                    bl = bl if bl is not None else 13
                    a = comp.color_heatmap(a, t, c, bl)
                else:
                    a = comp.color_colormap(a, colormap=c)
            else:
                a = comp.color_colormap(a, b, u)

            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class FindEdgeNode(JOVImageInOutBaseNode):
    NAME = "🔳 Find Edges (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Find Edges on an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "low": ("FLOAT", {"default": 0.27, "min": 0, "max": 1, "step": 0.01}),
                "high": ("FLOAT", {"default": 0.72, "min": 0, "max": 1, "step": 0.01}),
            }}
        return deep_merge_dict(IT_PIXELS_REQUIRED, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            low: Optional[list[float]]=None,
            high: Optional[list[float]]=None,
            invert: Optional[list[float]]=None)  -> tuple[torch.Tensor, torch.Tensor]:

        low = low or [None]
        high = high or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, low, high, invert):
            image, lo, hi, i = data
            image = tensor2cv(image)

            lo = lo or 0.27
            hi = hi or 0.72

            image = comp.morph_edge_detect(image, low=lo, high=hi)

            if i != 0:
                image = comp.light_invert(image, i)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class HSVNode(JOVImageInOutBaseNode):
    NAME = "🌈 HSV (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Adjust the Hue, Saturation, Value, Contrast and Gamma of the input."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "hue": ("FLOAT",{"default": 0, "min": 0, "max": 1, "step": 0.01},),
                "saturation": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}, ),
                "value": ("FLOAT", {"default": 1, "min": 0, "max": 250, "step": 0.01}, ),
                "contrast": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}, ),
                "gamma": ("FLOAT", {"default": 1, "min": 0, "max": 250, "step": 0.01}, ),
            }}
        return deep_merge_dict(IT_PIXELS_REQUIRED, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            hue: Optional[list[float]]=None,
            saturation: Optional[list[float]]=None,
            value: Optional[list[float]]=None,
            contrast: Optional[list[float]]=None,
            gamma: Optional[list[float]]=None,
            invert: Optional[list[float]]=None) -> tuple[torch.Tensor, torch.Tensor]:

        hue = hue or [None]
        saturation = saturation or [None]
        value = value or [None]
        contrast = contrast or [None]
        gamma = gamma or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, hue, saturation, value, contrast, gamma, invert):

            img, h, s, v, c, g, i = data
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

            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class LevelsNode(JOVImageInOutBaseNode):
    NAME = "🛗 Level Adjust (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Clip an input based on a low, high and mid point value"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "low": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01},),
                "mid": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01},),
                "high": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01},),
                "gamma": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01},),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            low: Optional[list[float]]=None,
            mid: Optional[list[float]]=None,
            high: Optional[list[float]]=None,
            gamma: Optional[list[float]]=None,
            invert: Optional[list[float]]=None)  -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for data in zip_longest_fill(pixels, low, mid, high, gamma, invert):
            img, l, m, h, g, i = data

            # img = tensor2pil(img)
            l = l or 0
            m = m or 0.5
            h = h or 1
            g = g or 1
            i = i or 0

            img = torch.maximum(img - l, torch.tensor(0.0))
            img = torch.minimum(img, (h - l))
            img = (img + m) - 0.5
            img = torch.sign(img) * torch.pow(torch.abs(img), 1.0 / g)
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
    NAME = "📉 Threshold (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Clip an input to explicit 0 or 1"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "adapt": ( EnumThresholdAdapt._member_names_, {"default": EnumThresholdAdapt.ADAPT_NONE.name}),
                "op": ( EnumThreshold._member_names_, {"default": EnumThreshold.BINARY.name}),
                "threshold": ("FLOAT", {"default": 0.5, "min": -100, "max": 100, "step": 0.01},),
                "block": ("INT", {"default": 3, "min": 3, "max": 103, "step": 1},),
            }}
        return deep_merge_dict(IT_PIXELS_REQUIRED, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            adapt: Optional[list[str]]=None,
            op: Optional[list[str]]=None,
            threshold: Optional[list[float]]=None,
            block: Optional[list[int]]=None,
            invert: Optional[list[float]]=None)  -> tuple[torch.Tensor, torch.Tensor]:

        op = op or [None]
        adapt = adapt or [None]
        threshold = threshold or [None]
        block = block or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, op, adapt, threshold, block, invert):
            img, o, a, t, b, i = data
            img = tensor2cv(img)

            o = EnumThreshold[o].value if o is not None else EnumThreshold.BINARY
            a = EnumThresholdAdapt[a].value if a is not None else EnumThresholdAdapt.ADAPT_NONE
            t = t if t is not None else 0.5
            b = b if b is not None else 3
            i = i if i is not None else 0

            img = comp.adjust_threshold(img, threshold=t, mode=o, adapt=a, block=b, const=t)
            if i != 0:
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
