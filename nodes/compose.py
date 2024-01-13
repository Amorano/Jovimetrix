"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition
"""

import cv2
import torch
import numpy as np
from loguru import logger

import comfy

from Jovimetrix import JOVImageInOutBaseNode, \
    IT_PIXELS, IT_RGB, IT_PIXEL_MASK, IT_INVERT, IT_REQUIRED, IT_RGBA_IMAGE, \
    MIN_IMAGE_SIZE, IT_TRANS, IT_ROT, IT_SCALE, IT_PIXEL2

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import parse_number, parse_tuple, zip_longest_fill, \
    deep_merge_dict,\
    EnumTupleType

from Jovimetrix.sup.comp import geo_rotate, geo_translate, comp_blend, geo_crop_polygonal, \
    geo_edge_wrap, geo_scalefit, geo_mirror, light_invert, \
    EnumScaleMode, EnumInterpolation, EnumBlendType

from Jovimetrix.sup.image import image_merge, image_split, tensor2cv, cv2tensor, \
    cv2mask, pixel_convert, image_stack, \
    EnumEdge, EnumMirrorMode, EnumOrientation, \
    IT_WHMODE, IT_SAMPLE

from Jovimetrix.sup.color import color_theory, EnumColorTheory

from Jovimetrix.sup.mapping import remap_fisheye, remap_perspective, remap_polar, \
    remap_sphere, EnumProjection

# =============================================================================

class TransformNode(JOVImageInOutBaseNode):
    NAME = "TRANSFORM (JOV) 🏝️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Translate, Rotate, Scale, Tile, Mirror, Re-project and invert an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
                Lexicon.MIRROR: (EnumMirrorMode._member_names_, {"default": EnumMirrorMode.NONE.name}),
                Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "max": 1, "min": "0", "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.TILE: ("VEC2", {"default": (1, 1), "step": 1, "min": 1, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
                Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
                Lexicon.PROJECTION: (EnumProjection._member_names_, {"default": EnumProjection.NORMAL.name}),
                Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "precision": 4, "step": 0.005})
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_TRANS, IT_ROT, IT_SCALE, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        offset = parse_tuple(Lexicon.OFFSET, kw, typ=EnumTupleType.FLOAT, default=(0., 0.,), clip_min=-1, clip_max=1)
        angle = kw.get(Lexicon.ANGLE, [0])
        size = parse_tuple(Lexicon.SIZE, kw, typ=EnumTupleType.FLOAT, default=(1., 1.,), zero=0.001)
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        mirror = kw.get(Lexicon.MIRROR, [EnumMirrorMode.NONE])
        mirror_pivot = parse_tuple(Lexicon.PIVOT, kw, typ=EnumTupleType.FLOAT, default=(0.5, 0.5,), clip_min=0, clip_max=1)
        tile_xy = parse_tuple(Lexicon.TILE, kw, default=(1, 1,), clip_min=1)
        proj = kw.get(Lexicon.PROJECTION, [EnumProjection.NORMAL])
        strength = kw.get(Lexicon.STRENGTH, [1])
        tltr = parse_tuple(Lexicon.TLTR, kw, EnumTupleType.FLOAT, (0, 0, 1, 1,), 0, 1)
        blbr = parse_tuple(Lexicon.BLBR, kw, EnumTupleType.FLOAT, (0, 0, 1, 1,), 0, 1)
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])
        res = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], 0, 1)
        params = [tuple(x) for x in zip_longest_fill(pixels, offset, angle, size, edge, wihi, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, res, i)]
        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, offset, angle, size, edge, wihi, tile_xy, mirror, mirror_pivot, pr, str, tltr, blbr, mode, res, i) in enumerate(params):
            w, h = wihi
            if img is not None:
                img = tensor2cv(img)
                sX, sY = size
                if sX < 0:
                    img = cv2.flip(img, 1)
                    sX = -sX
                if sY < 0:
                    img = cv2.flip(img, 0)
                    sY = -sY

                # SCALE
                res = EnumInterpolation[res]
                if sX != 1. or sY != 1.:
                    wx =  int(max(1, w * sX))
                    hx =  int(max(1, h * sY))
                    img = cv2.resize(img, (wx, hx), interpolation=res.value)

                # ROTATION
                if angle != 0:
                    img = geo_rotate(img, angle)

                # TRANSLATION
                oX, oY = offset
                if oX != 0. or oY != 0.:
                    img = geo_translate(img, oX, oY)

                if edge != "CLIP":
                    tx = ty = 0
                    if edge in ["WRAP", "WRAPX"] and sX < 1.:
                        tx = 1. / sX - 1

                    if edge in ["WRAP", "WRAPY"] and sY < 1.:
                        ty = 1. / sY - 1

                    img = geo_edge_wrap(img, tx, ty)
                    # h, w = img.shape[:2]

                # MIRROR
                mirror = EnumMirrorMode[mirror].name
                mpx, mpy = mirror_pivot
                if 'X' in mirror:
                    img = geo_mirror(img, mpx, 1, invert=i)
                if 'Y' in mirror:
                    img = geo_mirror(img, mpy, 0, invert=i)

                # TILE
                tx, ty = tile_xy
                if (tx := int(tx)) > 1 or (ty := int(ty)) > 1:
                    img = geo_edge_wrap(img, tx - 1, ty - 1)
                    img = geo_scalefit(img, w, h, EnumScaleMode.FIT)

                # RE-PROJECTION
                x1, y1, x2, y2 = tltr
                x4, y4, x3, y3 = blbr
                #sw, sh = w, h
                #if mode == EnumScaleMode.NONE:
                sh, sw = img.shape[:2]
                x1, x2, x3, x4 = map(lambda x: x * sw, [x1, x2, x3, x4])
                y1, y2, y3, y4 = map(lambda y: y * sh, [y1, y2, y3, y4])
                img = remap_perspective(img, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

                # RE-PROJECTION
                pr = EnumProjection[pr]
                match pr:
                    case EnumProjection.SPHERICAL:
                        img = remap_sphere(img, str)
                    case EnumProjection.FISHEYE:
                        img = remap_fisheye(img, str)
                    case EnumProjection.POLAR:
                        img = remap_polar(img)

                if i != 0:
                    img = light_invert(img, i)

                #img = geo_crop(img)
                img = geo_scalefit(img, w, h, mode, res)

            else:
                img = np.zeros((h, w, 3), dtype=np.uint8)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))
            pbar.update_absolute(idx)

        return (
            torch.stack(images),
            torch.stack(masks)
        )

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
        params = [tuple(x) for x in zip_longest_fill(pixelA, pixelB, mask, func, alpha, flip, wihi, mode, sample, i)]
        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pa, pb, ma, op, a, fl, wihi, sm, rs, i) in enumerate(params):
            w, h = wihi
            pa = tensor2cv(pa) if pa is not None else np.zeros((h, w, 3), dtype=np.uint8)
            pb = tensor2cv(pb) if pb is not None else np.zeros((h, w, 3), dtype=np.uint8)
            ma = tensor2cv(ma) if ma is not None else np.zeros((h, w), dtype=np.uint8)

            if fl:
                pa, pb = pb, pa

            op = EnumBlendType[op]
            img = comp_blend(pa, pb, ma, op, a)
            nh, nw = img.shape[:2]

            rs = EnumInterpolation[rs]
            if h != nh or w != nw:
                # logger.debug("{} {} {} {}", w, h, nw, nh)
                img = geo_scalefit(img, w, h, sm, rs)

            if i != 0:
                img = light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))
            pbar.update_absolute(idx)

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
            r, g, b, a = image_split(img)
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
        params = [tuple(x) for x in zip_longest_fill(R, G, B, A, wihi, mode, sample, i)]

        if len(R)+len(B)+len(G)+len(A) == 0:
            return (
                torch.stack([torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu")]),
                torch.stack([torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=torch.uint8, device="cpu")]),
            )

        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (r, g, b, a, wihi, m, rs, i) in enumerate(params):
            w, h = wihi
            r = tensor2cv(r) if r is not None else np.zeros((h, w, 3), dtype=np.uint8)
            g = tensor2cv(g) if g is not None else np.zeros((h, w, 3), dtype=np.uint8)
            b = tensor2cv(b) if b is not None else np.zeros((h, w, 3), dtype=np.uint8)
            a = tensor2cv(a) if a is not None else np.zeros((h, w, 3), dtype=np.uint8)
            rs = EnumInterpolation[rs]
            img = image_merge(r, g, b, a, w, h)
            img = geo_scalefit(img, w, h, m, rs)

            if i != 0:
                img = light_invert(img, i)
            images.append(cv2tensor(img))
            masks.append(cv2mask(img))
            pbar.update_absolute(idx)

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
        return deep_merge_dict(IT_REQUIRED, d, IT_PIXEL2, IT_WHMODE, IT_SAMPLE)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixelA = kw.get(Lexicon.PIXEL_A, [None])
        pixelB = kw.get(Lexicon.PIXEL_B, [None])
        axis = kw.get(Lexicon.AXIS, [EnumOrientation.GRID])
        stride = kw.get(Lexicon.STEP, [5])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        params = [tuple(x) for x in zip_longest_fill(pixelA, pixelB, axis, stride, wihi, mode, sample)]
        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (a, b, ax, st, wihi, m, rs) in enumerate(params):
            w, h = wihi
            a = tensor2cv(a) if a is not None else None
            b = tensor2cv(b) if a is not None else None
            a, b = pixel_convert(a, b)
            if a is None and b is None:
                images.append(torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu"))
                masks.append(torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=torch.uint8, device="cpu"))
                continue

            pixels = [a, b]
            # ma = tensor2cv(ma) if ma is not None else np.full((h, w), 255, dtype=np.uint8)
            rs = EnumInterpolation[rs]
            ax = EnumOrientation[ax]
            # color = 255
            img = image_stack(pixels, ax, st, mode=EnumScaleMode.FIT, sample=rs)
            if m != EnumScaleMode.NONE:
                img = geo_scalefit(img, w, h, m, rs)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))
            pbar.update_absolute(idx)

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
                Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
                Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, d, IT_RGB, IT_INVERT)

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        tltr = parse_tuple(Lexicon.TLTR, kw, EnumTupleType.FLOAT, (0, 0, 1, 1,), 0, 1)
        blbr = parse_tuple(Lexicon.BLBR, kw, EnumTupleType.FLOAT, (0, 0, 1, 1,), 0, 1)
        rgb = parse_tuple(Lexicon.RGB, kw, default=(0, 0, 0,), clip_min=0, clip_max=255)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], 0, 1)
        params = [tuple(x) for x in zip_longest_fill(pixels, tltr, blbr, rgb, i)]
        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, tltr, blbr, rgb, i) in enumerate(params):
            if img is None:
                images.append(torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu"))
                masks.append(torch.ones((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=torch.uint8, device="cpu"))
                continue

            img = tensor2cv(img)
            x1, y1, x2, y2 = tltr
            x4, y4, x3, y3 = blbr
            img, mask = geo_crop_polygonal(img, (x1, y1), (x2, y2), (x3, y3), (x4, y4), rgb)

            if i != 0:
                img = light_invert(img, i)
                mask = light_invert(mask, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(mask))
            pbar.update_absolute(idx)

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
        params = [tuple(x) for x in zip_longest_fill(pixels, scheme, user, i)]
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, s, user, i) in enumerate(params):
            img = tensor2cv(img) if img is not None else np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8)
            a, b, c, d, e = color_theory(img, user, EnumColorTheory[s])
            if i != 0:
                a = light_invert(a, i)
                b = light_invert(b, i)
                c = light_invert(c, i)
                d = light_invert(d, i)
                e = light_invert(e, i)

            imageA.append(cv2tensor(a))
            imageB.append(cv2tensor(b))
            imageC.append(cv2tensor(c))
            imageD.append(cv2tensor(d))
            imageE.append(cv2tensor(e))

            pbar.update_absolute(idx)

        return (
            torch.stack(imageA),
            torch.stack(imageB),
            torch.stack(imageC),
            torch.stack(imageD),
            torch.stack(imageE)
        )
