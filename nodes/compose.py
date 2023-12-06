"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition
"""

from typing import Optional

import torch
import numpy as np

from Jovimetrix.jovimetrix import tensor2cv, cv2tensor, cv2mask, \
    zip_longest_fill, deep_merge_dict, \
    JOVImageInOutBaseNode, Logger, \
    IT_PIXELS, IT_COLOR, IT_WH, IT_WHMODE, WILDCARD, IT_PIXEL2, IT_INVERT, \
    IT_REQUIRED, IT_WHMODE

from Jovimetrix.sup.comp import geo_crop, geo_scalefit, comp_blend, light_invert, \
    pixel_split, pixel_merge, image_stack, \
    BlendType, EnumInterpolation, EnumOrientation, EnumScaleMode, \
    EnumBlendType, \
    IT_SAMPLE

# =============================================================================

class BlendNode(JOVImageInOutBaseNode):
    NAME = "⚗️ Blend (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)."
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "func": (EnumBlendType, {"default": EnumBlendType[0]}),
                "alpha": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            },
            "optional": {
                "flip": ("BOOLEAN", {"default": False}),
                "mask": (WILDCARD, {})
        }}
        return deep_merge_dict(IT_PIXEL2, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self,
            pixelA: Optional[list[torch.tensor]]=None,
            pixelB: Optional[list[torch.tensor]]=None,
            mask: Optional[list[torch.tensor]]=None,
            func: Optional[list[str]]=None,
            alpha: Optional[list[float]]=None,
            flip: Optional[list[bool]]=None,
            width: Optional[list[int]]=None,
            height: Optional[list[int]]=None,
            mode: Optional[list[str]]=None,
            resample: Optional[list[str]]=None,
            invert: Optional[list[float]]=None,
            ) -> tuple[torch.Tensor, torch.Tensor]:

        pixelA = pixelA or [None]
        pixelB = pixelB or [None]
        mask = mask or [None]
        func = func or [None]
        alpha = alpha or [None]
        flip = flip or [None]
        width = width or [None]
        height = height or [None]
        mode = mode or [None]
        resample = resample or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, mask, func, alpha, flip,
                                     width, height, mode, resample, invert):

            pa, pb, ma, f, a, fl, w, h, sm, rs, i = data
            pa = tensor2cv(pa) if pa is not None else np.zeros((h, w, 4), dtype=np.uint8)
            pb = tensor2cv(pb) if pb is not None else np.zeros((h, w, 4), dtype=np.uint8)
            ma = tensor2cv(ma) if ma is not None else np.zeros((h, w), dtype=np.uint8)

            if fl:
                pa, pb = pb, pa

            f = BlendType[f] if f is not None else BlendType.NORMAL
            image = comp_blend(pa, pb, ma, f, a)

            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
            image = geo_scalefit(image, w, h, sm, rs)

            if i != 0:
                image = light_invert(image, i)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class PixelSplitNode(JOVImageInOutBaseNode):
    NAME = "💔 Pixel Split (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "SPLIT THE R-G-B from an image"

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "IMAGE", "MASK",)
    RETURN_NAMES = ("❤️", "🟥", "💚", "🟩", "💙", "🟦")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS)

    def run(self, pixels: list[torch.tensor])  -> tuple[torch.Tensor, torch.Tensor]:
        ret = {
            'r': [],
            'g': [],
            'b': [],
            'a': [],
            'rm': [],
            'gm': [],
            'bm': [],
            'ba': [],
        }

        for image in pixels:
            image = tensor2cv(image)
            image, mask = pixel_split(image)
            r, g, b = image
            ret['r'].append(r)
            ret['g'].append(g)
            ret['b'].append(b)

            r, g, b = mask
            ret['rm'].append(r)
            ret['gm'].append(g)
            ret['bm'].append(b)

        return (
            torch.stack(ret['r']),
            torch.stack(ret['rm']),
            torch.stack(ret['g']),
            torch.stack(ret['gm']),
            torch.stack(ret['b']),
            torch.stack(ret['bm']),
        )

class PixelMergeNode(JOVImageInOutBaseNode):
    NAME = "🫱🏿‍🫲🏼 Pixel Merge (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Merge 3/4 single channel inputs to make an image."

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("🖼️", "😷", )
    OUTPUT_IS_LIST = (True, True, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "R": (WILDCARD, {}),
                "G": (WILDCARD, {}),
                "B": (WILDCARD, {}),
            }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self,
            width:int,
            height:int,
            mode:str,
            resample: list[str],
            invert:float,
            R: Optional[list[torch.tensor]]=None,
            G: Optional[list[torch.tensor]]=None,
            B: Optional[list[torch.tensor]]=None)  -> tuple[torch.Tensor, torch.Tensor]:

        R = R or [None]
        G = G or [None]
        B = B or [None]

        if len(R)+len(B)+len(G) == 0:
            zero = cv2tensor(np.zeros([height[0], width[0], 3], dtype=np.uint8))
            return (
                torch.stack([zero]),
                torch.stack([zero]),
            )

        masks = []
        images = []
        for data in zip_longest_fill(R, G, B, width, height, mode, resample, invert):
            r, g, b, w, h, m, rs, i = data

            x = b if b is not None else g if g is not None else r if r is not None else None
            if x is None:
                Logger.err(self.NAME, "no images to process")
                continue

            _h, _w = x.shape[:2]
            w = w or _w
            h = h or _h
            empty = np.full((h, w), 0, dtype=np.uint8)
            r = tensor2cv(r) if r is not None else empty
            g = tensor2cv(g) if g is not None else empty
            b = tensor2cv(b) if b is not None else empty
            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4

            image = pixel_merge(b, g, r, None, w, h, m, rs)

            if i != 0:
                image = light_invert(image, i)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks),
        )

class MergeNode(JOVImageInOutBaseNode):
    NAME = "➕ Merge (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Union multiple latents horizontal, vertical or in a grid."
    SORT = 15

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "axis": (EnumOrientation._member_names_, {"default": EnumOrientation.GRID.name}),
                "stride": ("INT", {"min": 1, "step": 1, "default": 5}),
            },
            "optional": {
                "matte": (WILDCARD, {}),
            }}
        return deep_merge_dict(IT_PIXEL2, d, IT_WHMODE, IT_SAMPLE)

    def run(self,
            pixelA:Optional[list[torch.tensor]]=None,
            pixelB:Optional[list[torch.tensor]]=None,
            matte:Optional[list[torch.tensor]]=None,
            axis:Optional[list[str]]=None,
            stride:Optional[list[int]]=None,
            width:Optional[list[int]]=None,
            height:Optional[list[int]]=None,
            mode:Optional[list[str]]=None,
            resample:Optional[list[str]]=None,
            ) -> tuple[torch.Tensor, torch.Tensor]:

        pixelA = pixelA or [None]
        pixelB = pixelB or [None]
        matte = matte or [None]
        axis = axis or [None]
        stride = stride or [None]
        width = width or [None]
        height = height or [None]
        mode = mode or [None]
        resample = resample or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, matte, axis, stride, width, height, mode, resample):

            pa, pb, ma, ax, st, w, h, m, rs = data
            pixelA = pa or (torch.zeros((h, w, 3), dtype=torch.uint8),)
            pixelB = pb or (torch.zeros((h, w, 3), dtype=torch.uint8),)
            pixels = pa + pb
            pixels = [tensor2cv(image) for image in pixels]

            if ma is None:
                ma = np.zeros((h, w, 3), dtype=torch.uint8)
            else:
                ma = tensor2cv(ma)

            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
            ax = EnumOrientation[ax] if ax is not None else EnumOrientation.HORIZONTAL
            image = image_stack(pixels, ax, st, ma, EnumScaleMode.FIT, rs)

            if m != EnumScaleMode.NONE:
                image = geo_scalefit(image, w, h, m, rs)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class CropNode(JOVImageInOutBaseNode):
    NAME = "✂️ Crop (jov)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Robust cropping with color fill"
    SORT = 55

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "top": ("FLOAT", {"default": 0, "min": 0, "step": 0.01}),
                "left": ("FLOAT", {"default": 0, "min": 0, "step": 0.01}),
                "bottom": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
                "right": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            },
            "optional": {
                "pad":  ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_PIXELS, IT_WH, d, IT_COLOR,  IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            pad: Optional[list[bool]]=None,
            top: Optional[list[float]]=None,
            left: Optional[list[float]]=None,
            bottom: Optional[list[float]]=None,
            right: Optional[list[float]]=None,
            R: Optional[list[float]]=None,
            G: Optional[list[float]]=None,
            B: Optional[list[float]]=None,
            width: Optional[list[int]]=None,
            height: Optional[list[int]]=None,
            invert: Optional[list[float]]=None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        pad = pad or [None]
        top = top or [None]
        left = left or [None]
        bottom = bottom or [None]
        right = right or [None]
        R = R or [None]
        G = G or [None]
        B = B or [None]
        width = width or [None]
        height = height or [None]
        invert = invert or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, pad, top, left, bottom, right,
                                     R, G, B, width, height, invert):

            image, p, t, l, b, r, _r, _g, _b, w, h, i = data

            image = tensor2cv(image)
            p = p or False
            t = t or 0
            l = l or 0
            b = b or 1
            r = r or 1
            color = (_r * 255, _g * 255, _b * 255)
            w = w or image.shape[1]
            h = h or image.shape[1]
            i = i or 0
            Logger.debug(self.NAME, l, t, r, b, w, h, p, color)

            image = geo_crop(image, l, t, r, b, w, h, p, color)
            if i != 0:
                image = light_invert(image, i)
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