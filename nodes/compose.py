"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition
"""

import cv2

import torch
import numpy as np

from Jovimetrix import parse_number, parse_tuple, tensor2cv, cv2tensor, cv2mask, \
    zip_longest_fill, deep_merge_dict, \
    EnumTupleType, JOVImageInOutBaseNode, Logger, Lexicon, \
    IT_PIXELS, IT_RGBA, IT_WH, IT_WHMODE, IT_PIXEL_MASK, IT_BBOX, IT_INVERT, \
    IT_REQUIRED, IT_RGBA_IMAGE, MIN_IMAGE_SIZE, MIN_IMAGE_SIZE

from Jovimetrix.sup import comp
from Jovimetrix.sup.comp import \
    EnumInterpolation, EnumOrientation, EnumScaleMode, EnumColorTheory, EnumBlendType, \
    IT_SAMPLE

# =============================================================================

class BlendNode(JOVImageInOutBaseNode):
    NAME = "BLEND (JOV) ⚗️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)."
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.FUNC: (EnumBlendType._member_names_, {"default": EnumBlendType.NORMAL.name}),
                Lexicon.A: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL_MASK, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixelA = kw.get(Lexicon.PIXEL_A, [None])
        pixelB = kw.get(Lexicon.PIXEL_B, [None])
        mask = kw.get(Lexicon.MASK, [None])
        func = kw.get(Lexicon.FUNC, [EnumBlendType.NORMAL])
        alpha = kw.get(Lexicon.A, [1])
        flip = kw.get(Lexicon.FLIP, [False])
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, mask, func, alpha, flip,
                                     wihi, mode, sample, i):

            pa, pb, ma, op, a, fl, wihi, sm, rs, i = data
            w, h = wihi
            pa = tensor2cv(pa) if pa is not None else np.zeros((h, w, 3), dtype=np.uint8)
            pb = tensor2cv(pb) if pb is not None else np.zeros((h, w, 3), dtype=np.uint8)
            ma = tensor2cv(ma) if ma is not None else np.zeros((h, w), dtype=np.uint8)

            if fl:
                pa, pb = pb, pa

            op = EnumBlendType[op]
            img = comp.comp_blend(pa, pb, ma, op, a)
            nh, nw = img.shape[:2]

            rs = EnumInterpolation[rs]
            if h != nh or w != nw:
                Logger.debug(self, w, h, nw, nh)
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
    RETURN_NAMES = (Lexicon.RI, Lexicon.GI, Lexicon.BI, Lexicon.MI, Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.M)
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
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

        pixels = kw.get(Lexicon.PIXEL, [None])
        for img in pixels:
            img = tensor2cv(img) if img  is not None else np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            r, g, b, a = comp.image_split(img)
            e = np.zeros((h, w), dtype=np.uint8)

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

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_RGBA_IMAGE, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        R = kw.get(Lexicon.R, [None])
        G = kw.get(Lexicon.G, [None])
        B = kw.get(Lexicon.B, [None])
        A = kw.get(Lexicon.A, [None])
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)

        if len(R)+len(B)+len(G)+len(A) == 0:
            zero = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8)
            return (
                torch.stack([zero]),
                torch.stack([zero]),
            )

        masks = []
        images = []
        for data in zip_longest_fill(R, G, B, A, wihi, mode, sample, i):
            r, g, b, a, wihi, m, rs, i = data
            w, h = wihi
            r = tensor2cv(r) if r is not None else np.zeros((h, w, 3), dtype=np.uint8)
            g = tensor2cv(g) if g is not None else np.zeros((h, w, 3), dtype=np.uint8)
            b = tensor2cv(b) if b is not None else np.zeros((h, w, 3), dtype=np.uint8)
            a = tensor2cv(a) if a is not None else np.zeros((h, w, 3), dtype=np.uint8)
            rs = EnumInterpolation[rs]
            img = comp.image_merge(r, g, b, a, w, h, m, rs)
            if i != 0:
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
        d = {"optional": {
                Lexicon.AXIS: (EnumOrientation._member_names_, {"default": EnumOrientation.GRID.name}),
                Lexicon.STEP: ("INT", {"min": 1, "step": 1, "default": 5}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL_MASK, d, IT_WHMODE, IT_SAMPLE)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixelA = kw.get(Lexicon.PIXEL_A, [None])
        pixelB = kw.get(Lexicon.PIXEL_B, [None])
        mask = kw.get(Lexicon.MASK, [None])
        axis = kw.get(Lexicon.AXIS, [EnumOrientation.GRID])
        stride = kw.get(Lexicon.STEP, [5])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        masks = []
        images = []
        for data in zip_longest_fill(pixelA, pixelB, mask, axis, stride, wihi, mode, sample):
            a, b, ma, ax, st, wihi, m, rs = data
            w, h = wihi
            a, b = comp.pixel_convert(a, b)
            if a is None and b is None:
                zero = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue

            pixels = [tensor2cv(a), tensor2cv(b)]
            ma = tensor2cv(ma) if ma is not None else np.zeros((h, w, 3), dtype=np.uint8)
            rs = EnumInterpolation[rs]
            ax = EnumOrientation[ax]
            img = comp.image_stack(pixels, ax, st, ma, EnumScaleMode.FIT, rs)
            if m != EnumScaleMode.NONE:
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
                Lexicon.PAD:  ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_BBOX, d, IT_RGBA, IT_WH, IT_INVERT)

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        bbox = parse_tuple(Lexicon.BBOX, kw, default=(0, 0, 1, 1,), clip_min=0, clip_max=1)
        pad = kw.get(Lexicon.PAD, [0])
        rgba = parse_tuple(Lexicon.RGBA, kw, default=(0, 0, 0, 255,), clip_min=0, clip_max=255)
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        masks = []
        images = []
        for img, p, bbox, rgba, wihi, i in zip_longest_fill(pixels, pad, bbox, rgba, wihi, i):
            t, l, b, r = bbox
            w, h = wihi

            if img is None:
                zero = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8)
                images.append(zero)
                masks.append(zero)
                continue
            img = tensor2cv(img)
            # rgb = (_r * 255, _g * 255, _b * 255)
            w = w or img.shape[1]
            h = h or img.shape[0]
            # Logger.debug(l, t, r, b, w, h, p, c)
            img = comp.geo_crop(img, l or 0, t or 0, r or 1, b or 1, w, h, p or False, rgba)
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
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (Lexicon.C1, Lexicon.C2, Lexicon.C3, Lexicon.C4, Lexicon.C5)
    OUTPUT_IS_LIST = (True, True, True, True, True)
    SORT = 65

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.SCHEME: (EnumColorTheory._member_names_, {"default": EnumColorTheory.COMPLIMENTARY.name}),
                Lexicon.VALUE: ("INT", {"default": 45, "min": -90, "max": 90, "step": 1})
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_INVERT)

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        imageA = []
        imageB = []
        imageC = []
        imageD = []
        imageE = []
        pixels = kw.get(Lexicon.PIXEL, [None])
        scheme = kw.get(Lexicon.SCHEME, [EnumColorTheory.COMPLIMENTARY])
        user = parse_number(Lexicon.VALUE, kw, EnumTupleType.INT, [0], clip_min=-180, clip_max=180)
        # kw.get(Lexicon.VALUE, [0])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)

        for img, s, user, i in zip_longest_fill(pixels, scheme, user, i):
            img = tensor2cv(img) if img is not None else np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8)
            a, b, c, d, e = comp.color_theory(img, user, EnumColorTheory[s])
            if i != 0:
                a = comp.light_invert(a, i)
                b = comp.light_invert(b, i)
                c = comp.light_invert(c, i)
                d = comp.light_invert(d, i)
                e = comp.light_invert(e, i)

            imageA.append(cv2tensor(a))
            imageB.append(cv2tensor(b))
            imageC.append(cv2tensor(c))
            imageD.append(cv2tensor(d))
            imageE.append(cv2tensor(e))

        return (
            torch.stack(imageA),
            torch.stack(imageB),
            torch.stack(imageC),
            torch.stack(imageD),
            torch.stack(imageE)
        )
