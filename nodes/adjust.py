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
    IT_PIXELS, IT_WHMODE, IT_PIXEL2, IT_INVERT, IT_WHMODE

from Jovimetrix.sup.comp import light_contrast, light_gamma, light_exposure, light_invert, \
    adjust_sharpen, adjust_threshold, morph_edge_detect, morph_emboss, color_colormap, color_heatmap, \
    EnumAdjustOP, EnumThresholdAdapt, EnumColorMap, EnumThreshold

# =============================================================================

class HSVNode(JOVImageInOutBaseNode):
    NAME = "🌈 HSV (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Adjust Hue, Saturation, Value, Contrast Gamma of input."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "hue": ("FLOAT",{"default": 0, "min": 0, "max": 1, "step": 0.01},),
                "saturation": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}, ),
                "value": ("FLOAT", {"default": 1, "min": 0, "max": 250, "step": 0.01}, ),
                "contrast": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}, ),
                "gamma": ("FLOAT", {"default": 1, "min": 0, "max": 250, "step": 0.01}, ),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            hue: Optional[list[float]]=None,
            saturation: Optional[list[float]]=None,
            value: Optional[list[float]]=None,
            contrast: Optional[list[float]]=None,
            gamma: Optional[list[float]]=None,
            invert: Optional[list[float]]=None) -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for idx, img in enumerate(pixels):
            img = tensor2cv(img)

            h = hue[min(idx, len(hue)-1)]
            s = saturation[min(idx, len(saturation)-1)]

            if h != 0 or s != 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                if h != 0:
                    h *= 255
                    img[:, :, 0] = (img[:, :, 0] + h) % 180

                if s != 1:
                    img[:, :, 1] = np.clip(img[:, :, 1] * s, 0, 255)

                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            if (val := contrast[min(idx, len(contrast)-1)]) != 0:
                img = light_contrast(img, 1 - val)

            if (val := value[min(idx, len(value)-1)]) != 1:
                img = light_exposure(img, val)

            if (val := gamma[min(idx, len(gamma)-1)]) != 1:
                img = light_gamma(img, val)

            if (val := invert[min(idx, len(invert)-1)]) != 0:
                img = light_invert(img, val)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class AdjustNode(JOVImageInOutBaseNode):
    NAME = "🕸️ Adjust (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/ADJUST"
    DESCRIPTION = "Find Edges, Blur, Sharpen and Emboss an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "func": (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name}),
            },
            "optional": {
                "radius": ("INT", {"default": 1, "min": 1,  "max": 2048, "step": 1}),
                "amount": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
                "low": ("FLOAT", {"default": 0.27, "min": 0, "max": 1, "step": 0.01}),
                "high": ("FLOAT", {"default": 0.72, "min": 0, "max": 1, "step": 0.01}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            func: Optional[list[str]]=None,
            radius: Optional[list[float]]=None,
            amount: Optional[list[float]]=None,
            low: Optional[list[float]]=None,
            high: Optional[list[float]]=None,
            invert: Optional[list[float]]=None)  -> tuple[torch.Tensor, torch.Tensor]:

        func = func or [None]
        radius = radius or [None]
        amount = amount or [None]
        low = low or [None]
        high = high or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, func, radius, amount, low, high, invert):
            image, op, rad, amt, lo, hi, i = data
            image = tensor2cv(image)

            op = op if op is not None else EnumAdjustOP.BLUR
            rad = rad if rad is not None else 3
            rad = rad if rad % 2 == 1 else rad + 1

            match op:
                case EnumAdjustOP.BLUR:
                    image = cv2.blur(image, (rad, rad))

                case EnumAdjustOP.STACK_BLUR:
                    image = cv2.stackBlur(image, (rad, rad))

                case EnumAdjustOP.GAUSSIAN_BLUR:
                    image = cv2.GaussianBlur(image, (rad, rad), sigmaX=float(amt))

                case EnumAdjustOP.MEDIAN_BLUR:
                    image = cv2.medianBlur(image, (rad, rad))

                case EnumAdjustOP.SHARPEN:
                    adjust_sharpen(image, kernel_size=rad, amount=amt)

                case EnumAdjustOP.EMBOSS:
                    image = morph_emboss(image, amt)

                case EnumAdjustOP.FIND_EDGES:
                    image = morph_edge_detect(image, low=lo, high=hi)

                case EnumAdjustOP.OUTLINE:
                    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, (rad, rad))

                case EnumAdjustOP.DILATE:
                    image = cv2.dilate(image, (rad, rad), iterations=int(amt))

                case EnumAdjustOP.ERODE:
                    image = cv2.erode(image, (rad, rad), iterations=int(amt))

                case EnumAdjustOP.OPEN:
                    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (rad, rad))

                case EnumAdjustOP.CLOSE:
                    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (rad, rad))

            Logger.debug(self.NAME, op, rad, amt, low, hi)

            if i != 0:
                image = light_invert(image, i)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

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
        d = {"required": {
                "op": ( EnumThreshold._member_names_, {"default": EnumThreshold.BINARY.name}),
                "adapt": ( EnumThresholdAdapt._member_names_, {"default": EnumThresholdAdapt.ADAPT_NONE.name}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01},),
                "block": ("INT", {"default": 3, "min": 1, "max": 101, "step": 1},),
                "const": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.01},),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            op: list[str],
            adapt: list[str],
            threshold: Optional[list[float]]=None,
            block: Optional[list[int]]=None,
            const: Optional[list[float]]=None,
            invert: Optional[list[float]]=None)  -> tuple[torch.Tensor, torch.Tensor]:

        op = op or [None]
        adapt = adapt or [None]
        threshold = threshold or [None]
        block = block or [None]
        const = const or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, op, adapt, threshold, block, const, invert):
            image, o, a, t, b, c, i = data
            image = tensor2cv(image)

            # force block into odd
            if block % 2 == 0:
                block += 1

            o = EnumThreshold[o]
            a = EnumThresholdAdapt[a]
            t = t if t is not None else 0.5
            b = b if b is not None else 3
            c = c if c is not None else 0
            i = i if i is not None else 0

            image = adjust_threshold(image, t, o, a, b, c)
            if i != 0:
                image = light_invert(image, i)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

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
        d = {"required": {
                "low": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01},),
                "mid": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01},),
                "high": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01},),
                "gamma": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01},),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODE, IT_INVERT)

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
            image, l, m, h, g, i = data

            l = l or 0
            m = m or 0.5
            h = h or 1
            g = g or 1
            i = i or 0

            # image = tensor2pil(image)
            image = torch.maximum(image - l, torch.tensor(0.0))
            image = torch.minimum(image, (h - l))
            image = (image + m) - 0.5
            image = torch.sign(image) * torch.pow(torch.abs(image), 1.0 / g)
            image = (image + 0.5) / h

            if i != 0:
                image = 1 - i - image
                # image = light_invert(image, i)

            images.append(image)
            masks.append(image)

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class ColorCNode(JOVImageInOutBaseNode):
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
        return deep_merge_dict(IT_PIXEL2, d)

    def run(self,
            pixelA: list[torch.tensor],
            pixelB: Optional[list[torch.tensor]]=None,
            colormap: Optional[list[str]]=None,
            usemap: Optional[list[bool]]=None,
            threshold: Optional[list[float]]=None,
            blur: Optional[list[float]]=None,
            flip: Optional[list[bool]]=None) -> tuple[torch.Tensor, torch.Tensor]:

        pixelB = pixelB or [None]
        colormap = colormap or [None]
        usemap = usemap or [None]
        threshold = threshold or [None]
        blur = blur or [None]
        flip = flip or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, colormap,
                                     usemap, threshold, blur, flip):

            a, b, c, u, t, bl, f = data
            a = tensor2cv(a)
            if b is not None:
                b = tensor2cv(b)

                if f is not None and f:
                    a, b = b, a

            if (u is not None and u):
                c = EnumColorMap[c]
                if t is not None and t != 0:
                    bl = bl if bl is not None else 13
                    image = color_heatmap(a, c, t, bl)
                image = color_colormap(a, None, c)
            else:
                image = color_colormap(a, b, u)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
