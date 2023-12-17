"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition
"""

from typing import Optional
import cv2

import torch
import numpy as np

from Jovimetrix import tensor2cv, cv2tensor, cv2mask, \
    zip_longest_fill, deep_merge_dict, \
    JOVImageInOutBaseNode, Logger, \
    IT_PIXELS, IT_COLOR, IT_WH, IT_WHMODE, WILDCARD, IT_PIXEL2, IT_INVERT, \
    IT_REQUIRED, IT_WHMODE, IT_PIXELS_REQUIRED

from Jovimetrix.sup import comp
from Jovimetrix.sup.comp import \
    BlendType, EnumInterpolation, EnumOrientation, EnumScaleMode, EnumColorTheory, \
    EnumBlendType, \
    IT_SAMPLE

# =============================================================================

class BlendNode(JOVImageInOutBaseNode):
    NAME = "BLEND (JOV) ⚗️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)."
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                "func": (EnumBlendType._member_names_, {"default": EnumBlendType.NORMAL.name}),
                "alpha": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            },
            "optional": {
                "flip": ("BOOLEAN", {"default": False}),
                "mask": (WILDCARD, {})
        }}
        return deep_merge_dict(IT_PIXEL2, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self,
            func: list[str],
            alpha: list[float],
            flip: list[bool],
            width: list[int],
            height: list[int],
            mode: list[str],
            resample: list[str],
            invert: list[float],
            pixelA: Optional[list[torch.tensor]]=None,
            pixelB: Optional[list[torch.tensor]]=None,
            mask: Optional[list[torch.tensor]]=None,
            ) -> tuple[torch.Tensor, torch.Tensor]:

        pixelA = pixelA or [None]
        pixelB = pixelB or [None]
        mask = mask or [None]
        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, mask, func, alpha, flip,
                                     width, height, mode, resample, invert):

            pa, pb, ma, f, a, fl, w, h, sm, rs, i = data
            pa = tensor2cv(pa) if pa is not None else np.zeros((h, w, 4), dtype=np.uint8)
            pb = tensor2cv(pb) if pb is not None else np.zeros((h, w, 4), dtype=np.uint8)
            ma = tensor2cv(ma) if ma is not None else np.zeros((h, w), dtype=np.uint8)

            if (fl or False):
                pa, pb = pb, pa

            f = EnumBlendType.NORMAL if f is None else EnumBlendType[f]
            img = comp.comp_blend(pa, pb, ma, f, a)

            nh, nw = img.shape[:2]
            rs = EnumInterpolation.LANCZOS4 if rs is None else EnumInterpolation[rs]
            if h != nh or w != nw:
                Logger.debug(w, h, nw, nh)
                img = comp.geo_scalefit(img, w, h, sm, rs)

            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class PixelSplitNode(JOVImageInOutBaseNode):
    NAME = "PIXEL SPLIT (JOV) 💔"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "SPLIT THE R-G-B-A from an image"

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "MASK", "MASK",)
    RETURN_NAMES = ("❤️", "💚", "💙", "🖤", "🟥", "🟩", "🟦", "⬛")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, )

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
            'am': [],
        }

        for img in pixels:
            img = tensor2cv(img)
            h, w = img.shape[:2]
            r, g, b, a = comp.image_split(img)
            e = np.full((h, w), 0, dtype=np.uint8)

            for rgb, mask, color in (
                ('r', 'rm', [e, e, r]),
                ('g', 'gm', [e, g, e]),
                ('b', 'bm', [b, e, e]),
                ('a', 'am', [a, a, a])):

                f = cv2.merge(color)
                ret[rgb].append(cv2tensor(f))
                ret[mask].append(cv2mask(f))

        return tuple(torch.stack(ret[key]) for key in ['r', 'g', 'b', 'a', 'rm', 'gm', 'bm', 'am'])

class PixelMergeNode(JOVImageInOutBaseNode):
    NAME = "PIXEL MERGE (JOV) 🫂"
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
                "A": (WILDCARD, {}),
            }}
        return deep_merge_dict(IT_REQUIRED, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self,
            width: list[int],
            height: list[int],
            mode: list[str],
            resample: list[str],
            invert: list[float],
            R: Optional[list[torch.tensor]]=None,
            G: Optional[list[torch.tensor]]=None,
            B: Optional[list[torch.tensor]]=None,
            A: Optional[list[torch.tensor]]=None)  -> tuple[torch.Tensor, torch.Tensor]:

        R = R or [None]
        G = G or [None]
        B = B or [None]
        A = A or [None]

        if len(R)+len(B)+len(G)+len(A) == 0:
            zero = cv2tensor(np.zeros([height[0], width[0], 3], dtype=np.uint8))
            return (
                torch.stack([zero]),
                torch.stack([zero]),
            )

        masks = []
        images = []
        for data in zip_longest_fill(R, G, B, A, width, height, mode, resample, invert):
            r, g, b, a, w, h, m, rs, i = data

            w = w or 0
            h = h or 0
            r = tensor2cv(r) if r is not None else None
            g = tensor2cv(g) if g is not None else None
            b = tensor2cv(b) if b is not None else None
            a = tensor2cv(a) if a is not None else None
            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
            img = comp.image_merge(r, g, b, a, w, h, m, rs)
            if (i or 0) != 0:
                img = comp.light_invert(img, i)
            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks),
        )

class MergeNode(JOVImageInOutBaseNode):
    NAME = "MERGE (JOV) ➕"
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
            axis:list[str],
            stride:list[int],
            width:list[int],
            height:list[int],
            mode:list[str],
            resample:list[str],
            pixelA:Optional[list[torch.tensor]]=None,
            pixelB:Optional[list[torch.tensor]]=None,
            matte:Optional[list[torch.tensor]]=None,
            ) -> tuple[torch.Tensor, torch.Tensor]:

        pixelA = pixelA or [None]
        pixelB = pixelB or [None]
        matte = matte or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, matte, axis, stride, width, height, mode, resample):

            pa, pb, ma, ax, st, w, h, m, rs = data
            pixelA = pa or (torch.zeros((h, w, 3), dtype=torch.uint8),)
            pixelB = pb or (torch.zeros((h, w, 3), dtype=torch.uint8),)
            pixels = pa + pb
            pixels = [tensor2cv(img) for img in pixels]

            ma = np.zeros((h, w, 3), dtype=torch.uint8) if ma is None else tensor2cv(ma)
            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
            ax = EnumOrientation[ax] if ax is not None else EnumOrientation.HORIZONTAL
            img = comp.image_stack(pixels, ax, st, ma, EnumScaleMode.FIT, rs)

            if (m or EnumScaleMode.NONE) != EnumScaleMode.NONE:
                img = comp.geo_scalefit(img, w, h, m, rs)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class CropNode(JOVImageInOutBaseNode):
    NAME = "CROP (JOV) ✂️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Robust cropping with color fill"
    SORT = 55

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "top": ("FLOAT", {"default": 0, "min": 0, "step": 0.01}),
                "left": ("FLOAT", {"default": 0, "min": 0, "step": 0.01}),
                "bottom": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
                "right": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            },
            "optional": {
                "pad":  ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_PIXELS_REQUIRED, IT_WH, d, IT_COLOR,  IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            pad: list[bool]=None,
            top: list[float]=None,
            left: list[float]=None,
            bottom: list[float]=None,
            right: list[float]=None,
            R: list[float]=None,
            G: list[float]=None,
            B: list[float]=None,
            width: list[int]=None,
            height: list[int]=None,
            invert: list[float]=None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        masks = []
        images = []
        for data in zip_longest_fill(pixels, pad, top, left, bottom, right,
                                     R, G, B, width, height, invert):

            img, p, t, l, b, r, _r, _g, _b, w, h, i = data

            img = tensor2cv(img)
            p = p or False
            t = t or 0
            l = l or 0
            b = b or 1
            r = r or 1
            color = (_r * 255, _g * 255, _b * 255)
            w = w or img.shape[1]
            h = h or img.shape[1]
            i = i or 0
            Logger.debug(l, t, r, b, w, h, p, color)

            img = comp.geo_crop(img, l, t, r, b, w, h, p, color)
            if i != 0:
                img = comp.light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class ColorTheoryNode(JOVImageInOutBaseNode):
    NAME = "COLOR THEORY (JOV) 🛞"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Re-project an input into various color theory mappings"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("🔵", "🟡", "🟣", "⚪")
    OUTPUT_IS_LIST = (True, True, True, True, )
    SORT = 65

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                "scheme": (EnumColorTheory._member_names_, {"default": EnumColorTheory.COMPLIMENTARY.name}),
            }}
        return deep_merge_dict(IT_PIXELS_REQUIRED, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            scheme: list[str],
            invert: list[float]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        imageA = []
        imageB = []
        imageC = []
        imageD = []

        for data in zip_longest_fill(pixels, scheme, invert):
            img, s, i = data
            img = tensor2cv(img)
            s = EnumColorTheory.COMPLIMENTARY if s is None else EnumColorTheory[s]
            Logger.debug(s, i)

            a, b, c, d = comp.color_theory(img, s)
            if (i or 0) != 0:
                a = comp.light_invert(a, i)
                b = comp.light_invert(b, i)
                c = comp.light_invert(c, i)
                d = comp.light_invert(d, i)

            Logger.debug(a.shape, b.shape, c.shape, d.shape)
            imageA.append(cv2tensor(a))
            imageB.append(cv2tensor(b))
            imageC.append(cv2tensor(c))
            imageD.append(cv2tensor(d))

        return (
            torch.stack(imageA),
            torch.stack(imageB),
            torch.stack(imageC),
            torch.stack(imageD)
        )

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass