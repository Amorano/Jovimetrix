"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition
"""

from enum import Enum
import math
import cv2
import torch
import numpy as np
from loguru import logger

import comfy

from Jovimetrix import IT_MATTE, JOVImageSimple, JOVImageMultiple, \
    IT_PIXEL, IT_RGB, IT_PIXEL2_MASK, IT_INVERT, IT_REQUIRED, \
    IT_RGBA_IMAGE, MIN_IMAGE_SIZE, IT_TRANS, IT_ROT, IT_SCALE

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import parse_number, parse_tuple, zip_longest_fill, \
    deep_merge_dict,\
    EnumTupleType

from Jovimetrix.sup.image import channel_count, channel_merge, channel_solid, \
    image_convert, image_crop, image_crop_center, \
    image_mask, image_rotate, image_scale, \
    image_translate, image_split, mask2cv, pixel_eval, tensor2cv, \
    image_crop_polygonal, image_edge_wrap, image_scalefit, cv2tensor, \
    cv2mask, image_stack, image_mirror, image_blend, \
    color_theory, remap_fisheye, remap_perspective, remap_polar, \
    remap_sphere, image_invert, \
    EnumImageType, EnumColorTheory, EnumProjection, \
    EnumScaleMode, EnumInterpolation, EnumBlendType, \
    EnumEdge, EnumMirrorMode, EnumOrientation, \
    IT_WHMODE

# =============================================================================

class TransformNode(JOVImageMultiple):
    NAME = "TRANSFORM (JOV) 🏝️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Translate, Rotate, Scale, Tile, Mirror, Re-project and invert an input."
    SORT = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.TILE: ("VEC2", {"default": (1, 1), "step": 1, "min": 1, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
            Lexicon.MIRROR: (EnumMirrorMode._member_names_, {"default": EnumMirrorMode.NONE.name}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "max": 1, "min": 0, "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.PROJECTION: (EnumProjection._member_names_, {"default": EnumProjection.NORMAL.name}),
            Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
            Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
            Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "precision": 4, "step": 0.005})
        }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL, IT_TRANS, IT_ROT, IT_SCALE, d, IT_WHMODE, IT_MATTE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        img = kw.get(Lexicon.PIXEL, [None])
        offset = parse_tuple(Lexicon.XY, kw, typ=EnumTupleType.FLOAT, default=(0., 0.,), clip_min=-1, clip_max=1)
        angle = kw.get(Lexicon.ANGLE, [0])
        size = parse_tuple(Lexicon.SIZE, kw, typ=EnumTupleType.FLOAT, default=(1., 1.,), zero=0.001)
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        mirror = kw.get(Lexicon.MIRROR, [EnumMirrorMode.NONE])
        mirror_pivot = parse_tuple(Lexicon.PIVOT, kw, typ=EnumTupleType.FLOAT, default=(0.5, 0.5,), clip_min=0, clip_max=1)
        tile_xy = parse_tuple(Lexicon.TILE, kw, default=(1, 1), clip_min=1)
        proj = kw.get(Lexicon.PROJECTION, [EnumProjection.NORMAL])
        tltr = parse_tuple(Lexicon.TLTR, kw, EnumTupleType.FLOAT, (0, 0, 1, 0,), 0, 1)
        blbr = parse_tuple(Lexicon.BLBR, kw, EnumTupleType.FLOAT, (0, 1, 1, 1,), 0, 1)
        strength = kw.get(Lexicon.STRENGTH, [1])
        mode = kw.get(Lexicon.MODE,[EnumScaleMode.NONE])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        color = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0), clip_min=1)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], 0, 1)
        params = [tuple(x) for x in zip_longest_fill(img, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, color, i)]
        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, color, i) in enumerate(params):

            img, mask = tensor2cv(img)
            cc, w, h = channel_count(img)[:3]
            if cc != 4:
                img = image_convert(img, 4)
            img[:,:,3] = mask

            sX, sY = size
            if sX < 0:
                img = cv2.flip(img, 1)
                sX = -sX

            if sY < 0:
                img = cv2.flip(img, 0)
                sY = -sY

            edge = EnumEdge[edge]
            if sX != 1. or sY != 1.:
                img = image_scale(img, (sX, sY))

            if offset[0] != 0. or offset[1] != 0.:
                img = image_translate(img, offset, edge)

            if angle != 0:
                img = image_rotate(img, angle, edge=edge)

            mirror = EnumMirrorMode[mirror]
            mpx, mpy = mirror_pivot
            img = image_mirror(img, mirror, mpx, mpy)

            color = pixel_eval(color, mode=EnumImageType.BGRA)
            tx, ty = tile_xy
            if (tx := int(tx)) > 1 or (ty := int(ty)) > 1:
                img = image_edge_wrap(img, tx / 2 - 0.5, ty / 2 - 0.5)
                img = image_scalefit(img, w, h, mode=EnumScaleMode.FIT)

            proj = EnumProjection[proj]
            match proj:
                case EnumProjection.PERSPECTIVE:
                    x1, y1, x2, y2 = tltr
                    x4, y4, x3, y3 = blbr
                    sh, sw = img.shape[:2]
                    x1, x2, x3, x4 = map(lambda x: x * sw, [x1, x2, x3, x4])
                    y1, y2, y3, y4 = map(lambda y: y * sh, [y1, y2, y3, y4])
                    img = remap_perspective(img, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                case EnumProjection.SPHERICAL:
                    img = remap_sphere(img, strength)
                case EnumProjection.FISHEYE:
                    img = remap_fisheye(img, strength)
                case EnumProjection.POLAR:
                    img = remap_polar(img)

            if proj != EnumProjection.NORMAL:
                img = image_scalefit(img, w, h, mode=EnumScaleMode.FIT)

            mode = EnumScaleMode[mode]
            if mode != EnumScaleMode.NONE:
                w, h = wihi
                sample = EnumInterpolation[sample]
                img = image_scalefit(img, w, h, mode, sample)

            mask = image_mask(img)
            if i != 0:
                img = image_invert(img, i)

            matte = channel_solid(w, h, color, chan=EnumImageType.RGB)
            img = image_blend(matte, img, mask)

            images.append(cv2tensor(img))
            masks.append(cv2mask(mask))
            pbar.update_absolute(idx)

        return images, masks,

class BlendNode(JOVImageSimple):
    NAME = "BLEND (JOV) ⚗️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)."
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.FUNC: (EnumBlendType._member_names_, {"default": EnumBlendType.NORMAL.name}),
                Lexicon.A: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL2_MASK, d, IT_WHMODE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixelA = kw.get(Lexicon.PIXEL_A, [None])
        pixelB = kw.get(Lexicon.PIXEL_B, [None])
        mask = kw.get(Lexicon.MASK, [None])
        func = kw.get(Lexicon.FUNC, [EnumBlendType.NORMAL])
        alpha = kw.get(Lexicon.A, [1])
        flip = kw.get(Lexicon.FLIP, [False])
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        color = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0), clip_min=1)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        params = [tuple(x) for x in zip_longest_fill(pixelA, pixelB, mask, func, alpha, flip, mode, wihi, sample, color, i)]
        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pa, pb, ma, op, alpha, fl, mode, wihi, sample, color, i) in enumerate(params):
            pa = tensor2cv(pa)[0] if pa is not None else None
            pb = tensor2cv(pb)[0] if pb is not None else None
            ma = tensor2cv(ma)[0] if ma is not None else None

            if fl:
                pa, pb = pb, pa

            op = EnumBlendType[op]
            mode = EnumScaleMode[mode]
            sample = EnumInterpolation[sample]
            # color = pixel_eval(color)
            img = image_blend(pa, pb, ma, op, alpha, color, mode, sample)
            width, height = wihi
            img = image_scalefit(img, width, height, mode, sample)
            if i != 0:
                img = image_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))
            pbar.update_absolute(idx)

        return images, masks,

class PixelSplitNode(JOVImageMultiple):
    NAME = "PIXEL SPLIT (JOV) 💔"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Splits images into constituent R, G and B and A channels."
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = (Lexicon.RI, Lexicon.GI, Lexicon.BI, Lexicon.MI)
    OUTPUT_IS_LIST = (True, True, True, True, )
    SORT = 40

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        ret = {
            'r': [],
            'g': [],
            'b': [],
            'a': [],
        }

        pixel = kw.get(Lexicon.PIXEL, [None])
        pbar = comfy.utils.ProgressBar(len(pixel))
        for idx, (img,) in enumerate(pixel):
            if img is None:
                img = np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8)
                mask = np.full((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), 255, dtype=np.uint8)
            else:
                img, mask = tensor2cv(img)

            h, w = img.shape[:2]
            r, g, b, _ = image_split(img)
            e = np.zeros((h, w), dtype=np.uint8)
            for rgb, color in (
                ('r', [e, e, r]),
                ('g', [e, g, e]),
                ('b', [b, e, e])):
                f = cv2.merge(color)
                ret[rgb].append(cv2tensor(f))

            ret['a'] = cv2mask(mask)
            pbar.update_absolute(idx)

        return ret['r'], ret['g'], ret['b'], ret['a']

class PixelMergeNode(JOVImageMultiple):
    NAME = "PIXEL MERGE (JOV) 🫂"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Combine 3 or 4 inputs into a single image."
    SORT = 45

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_RGBA_IMAGE)

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        R = kw.get(Lexicon.R, [None])
        G = kw.get(Lexicon.G, [None])
        B = kw.get(Lexicon.B, [None])
        A = kw.get(Lexicon.A, [None])
        params = [tuple(x) for x in zip_longest_fill(R, G, B, A)]

        if len(R)+len(B)+len(G)+len(A) == 0:
            e = torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu")
            m = torch.full((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), 255, dtype=torch.uint8, device="cpu")
            return [e], [m],

        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (r, g, b, a) in enumerate(params):
            r = tensor2cv(r)[0] if r is not None else None
            g = tensor2cv(g)[0] if g is not None else None
            b = tensor2cv(b)[0] if b is not None else None
            mask = tensor2cv(a)[0] if a is not None else None
            img = channel_merge([b, g, r, mask])
            images.append(cv2tensor(img))
            masks.append(a)
            pbar.update_absolute(idx)

        return images, masks,

class StackNode(JOVImageMultiple):
    NAME = "STACK (JOV) ➕"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Union multiple images horizontal, vertical or in a grid."
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False, False,)
    SORT = 55

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.AXIS: (EnumOrientation._member_names_, {"default": EnumOrientation.GRID.name}),
                Lexicon.STEP: ("INT", {"min": 0, "step": 1, "default": 0}),
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL, d, IT_WHMODE)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        axis = kw.get(Lexicon.AXIS, EnumOrientation.GRID)
        stride = kw.get(Lexicon.STEP, 0)
        mode = kw.get(Lexicon.MODE,EnumScaleMode.NONE)
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)[0]
        sample = kw.get(Lexicon.SAMPLE, EnumInterpolation.LANCZOS4)
        color = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0), clip_min=1)[0]
        color = pixel_eval(color)
        images = []
        for img in pixels:
            if img is not None:
                img, mask = tensor2cv(img)
                img = image_convert(img, 4)
            else:
                img = np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=np.uint8)
                mask = np.full((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), 255, dtype=np.uint8)
            img[:, :, 3] = mask
            images.append(img)

        axis = EnumOrientation[axis]
        img, mask = image_stack(images, axis, stride, color)

        mode = EnumScaleMode[mode]
        if mode != EnumScaleMode.NONE:
            sample = EnumInterpolation[sample]
            w, h = wihi
            img = image_scalefit(img, w, h, mode, sample)

        mask = img[:, :, 3][:, :]
        img = img[:, :, :3]

        matte = channel_solid(w, h, color, chan=EnumImageType.RGB)
        img = image_blend(matte, img, mask)
        return cv2tensor(img), cv2mask(mask),

class EnumCropMode(Enum):
    CENTER = 20
    CORNER = 0
    FREE = 10

class CropNode(JOVImageMultiple):
    NAME = "CROP (JOV) ✂️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Clip away sections of an image and backfill with optional color matte."
    SORT = 5

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.FUNC: (EnumCropMode._member_names_, {"default": EnumCropMode.CENTER.name}),
            Lexicon.XY: ("VEC2", {"default": (0, 0), "min": 0, "step": 1, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.WH: ("VEC2", {"default": (512, 512), "min": MIN_IMAGE_SIZE, "max": 8192, "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.TLTR: ("VEC4", {"default": (0, 0, 0, 1), "min": 0, "max": 1, "step": 0.01, "precision": 5, "round": 0.000001, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
            Lexicon.BLBR: ("VEC4", {"default": (1, 0, 1, 1), "min": 0, "max": 1, "step": 0.01, "precision": 5, "round": 0.000001, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
        }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL, d, IT_RGB)

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        func = kw.get(Lexicon.FUNC, [EnumCropMode.CENTER])
        # if less than 1 then use as scalar, over 1 = int(size)
        xy = parse_tuple(Lexicon.XY, kw, EnumTupleType.FLOAT, (0, 0,), 1)
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        tltr = parse_tuple(Lexicon.TLTR, kw, EnumTupleType.FLOAT, (0, 0, 0, 1,), 0, 1)
        blbr = parse_tuple(Lexicon.BLBR, kw, EnumTupleType.FLOAT, (1, 0, 1, 1,), 0, 1)
        color = parse_tuple(Lexicon.RGB, kw, default=(0, 0, 0,), clip_min=0, clip_max=255)
        params = [tuple(x) for x in zip_longest_fill(pixels, func, xy, wihi, tltr, blbr, color)]
        masks = []
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, func, xy, wihi, tltr, blbr, color) in enumerate(params):
            if img is None:
                images.append(torch.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=torch.uint8, device="cpu"))
                masks.append(torch.full((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), 255, dtype=torch.uint8, device="cpu"))
                continue

            img, _ = tensor2cv(img)
            width, height = wihi
            func = EnumCropMode[func]
            if func == EnumCropMode.FREE:
                y1, x1, y2, x2 = tltr
                y4, x4, y3, x3 = blbr
                points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                img = image_crop_polygonal(img, points)
            elif func == EnumCropMode.CORNER:
                img = image_crop(img, width, height, xy)
            else:
                img = image_crop_center(img, width, height)

            cc, w, h, chan = channel_count(img)
            mask = image_mask(img)
            if cc > 1:
                matte = channel_solid(w, h, color, chan=chan)
                img = image_blend(matte, img, mask)

            images.append(cv2tensor(img))
            masks.append(cv2mask(mask))
            pbar.update_absolute(idx)

        return images, masks,

class ColorTheoryNode(JOVImageMultiple):
    NAME = "COLOR THEORY (JOV) 🛞"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"
    DESCRIPTION = "Generate Complimentary, Triadic and Tetradic color sets."
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (Lexicon.C1, Lexicon.C2, Lexicon.C3, Lexicon.C4, Lexicon.C5)
    OUTPUT_IS_LIST = (True, True, True, True, True)
    SORT = 85

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.SCHEME: (EnumColorTheory._member_names_, {"default": EnumColorTheory.COMPLIMENTARY.name}),
                Lexicon.VALUE: ("INT", {"default": 45, "min": -90, "max": 90, "step": 1})
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL, d, IT_INVERT)

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
            img = tensor2cv(img)[0] if img is not None else np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8)
            a, b, c, d, e = color_theory(img, user, EnumColorTheory[s])
            if i != 0:
                a = image_invert(a, i)
                b = image_invert(b, i)
                c = image_invert(c, i)
                d = image_invert(d, i)
                e = image_invert(e, i)

            imageA.append(cv2tensor(a))
            imageB.append(cv2tensor(b))
            imageC.append(cv2tensor(c))
            imageD.append(cv2tensor(d))
            imageE.append(cv2tensor(e))

            pbar.update_absolute(idx)

        return imageA, imageB, imageC, imageD, imageE
