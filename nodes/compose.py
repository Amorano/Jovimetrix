"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition
"""

import cv2
import torch
from enum import Enum
from loguru import logger

import numpy as np

import comfy

from Jovimetrix import JOV_HELP_URL, WILDCARD, JOVImageMultiple, \
    IT_PIXEL, IT_RGB, IT_INVERT, IT_REQUIRED, \
    IT_RGBA_IMAGE, MIN_IMAGE_SIZE, IT_TRANS, IT_ROT, IT_SCALE

from Jovimetrix.sup.lexicon import Lexicon

from Jovimetrix.sup.util import parse_number, parse_tuple, zip_longest_fill, \
    deep_merge_dict,\
    EnumTupleType

from Jovimetrix.sup.image import batch_extract, channel_merge, \
    channel_solid, cv2tensor_full, \
    image_crop, image_crop_center, image_crop_polygonal, image_grayscale, \
    image_mask, image_mask_add, image_matte, image_rotate, image_scale, \
    image_translate, image_split, pixel_eval, tensor2cv, \
    image_edge_wrap, image_scalefit, cv2tensor, \
    image_stack, image_mirror, image_blend, \
    color_theory, remap_fisheye, remap_perspective, remap_polar, \
    remap_sphere, image_invert, \
    EnumImageType, EnumColorTheory, EnumProjection, \
    EnumScaleMode, EnumInterpolation, EnumBlendType, \
    EnumEdge, EnumMirrorMode, EnumOrientation, \
    IT_WHMODE

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"

# =============================================================================

class TransformNode(JOVImageMultiple):
    NAME = "TRANSFORM (JOV) 🏝️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Translate, Rotate, Scale, Tile, Mirror, Re-project and invert an input."
    SORT = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.TILE: ("VEC2", {"default": (1, 1), "step": 1, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
            Lexicon.MIRROR: (EnumMirrorMode._member_names_, {"default": EnumMirrorMode.NONE.name}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.PROJECTION: (EnumProjection._member_names_, {"default": EnumProjection.NORMAL.name}),
            Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "step": 0.005, "precision": 4, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
            Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "step": 0.005, "precision": 4, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
            Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "precision": 4, "step": 0.005}),
        }}
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL, IT_TRANS, IT_ROT, IT_SCALE, d, IT_WHMODE)
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-transform")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
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
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)
        params = [tuple(x) for x in zip_longest_fill(pA, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, matte)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pA, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, matte) in enumerate(params):

            matte = pixel_eval(matte, EnumImageType.BGRA)
            if pA is None:
                pA = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, matte, EnumImageType.BGRA)
                pA = cv2tensor_full(pA, matte)
                images.append(pA)
                pbar.update_absolute(idx)
                logger.debug("Should not be here")
                continue

            pA = tensor2cv(pA)
            sX, sY = size
            if sX < 0:
                pA = cv2.flip(pA, 1)
                sX = -sX

            if sY < 0:
                pA = cv2.flip(pA, 0)
                sY = -sY

            edge = EnumEdge[edge]
            sample = EnumInterpolation[sample]
            if sX != 1. or sY != 1.:
                pA = image_scale(pA, (sX, sY), sample, edge)

            if offset[0] != 0. or offset[1] != 0.:
                pA = image_translate(pA, offset, edge)

            if angle != 0:
                pA = image_rotate(pA, angle, edge=edge)

            mirror = EnumMirrorMode[mirror]
            if mirror != EnumMirrorMode.NONE:
                mpx, mpy = mirror_pivot
                pA = image_mirror(pA, mirror, mpx, mpy)

            tx, ty = tile_xy
            h, w = pA.shape[:2]
            if (tx := int(tx)) > 1 or (ty := int(ty)) > 1:
                pA = image_edge_wrap(pA, tx / 2 - 0.5, ty / 2 - 0.5)
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample, matte)

            proj = EnumProjection[proj]
            match proj:
                case EnumProjection.PERSPECTIVE:
                    x1, y1, x2, y2 = tltr
                    x4, y4, x3, y3 = blbr
                    sh, sw = pA.shape[:2]
                    x1, x2, x3, x4 = map(lambda x: x * sw, [x1, x2, x3, x4])
                    y1, y2, y3, y4 = map(lambda y: y * sh, [y1, y2, y3, y4])
                    pA = remap_perspective(pA, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                case EnumProjection.SPHERICAL:
                    pA = remap_sphere(pA, strength)
                case EnumProjection.FISHEYE:
                    pA = remap_fisheye(pA, strength)
                case EnumProjection.POLAR:
                    pA = remap_polar(pA)

            if proj != EnumProjection.NORMAL:
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample, matte)

            mode = EnumScaleMode[mode]
            if mode != EnumScaleMode.NONE:
                w, h = wihi
                pA = image_scalefit(pA, w, h, mode, sample, matte)

            pA = cv2tensor_full(pA, matte)
            images.append(pA)
            pbar.update_absolute(idx)
        return list(zip(*images))

class BlendNode(JOVImageMultiple):
    NAME = "BLEND (JOV) ⚗️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)."
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PIXEL_A: (WILDCARD, {"tooltip": "Background Plate"}),
            Lexicon.PIXEL_B: (WILDCARD, {"tooltip": "Image to Overlay on Background Plate"}),
            Lexicon.MASK: (WILDCARD, {"tooltip": "Optional Mask to use for Alpha Blend Operation. If empty, will use the ALPHA of B."}),
            Lexicon.FUNC: (EnumBlendType._member_names_, {"default": EnumBlendType.NORMAL.name, "tooltip": "Blending Operation"}),
            Lexicon.A: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Amount of Blending to Perform on the Selected Operation"}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
        }}
        d = deep_merge_dict(IT_REQUIRED, d, IT_INVERT, IT_WHMODE)
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-blend")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL_A, None)
        pA = [None] if pA is None else batch_extract(pA)
        pB = kw.get(Lexicon.PIXEL_B, None)
        pB = [None] if pB is None else batch_extract(pB)
        mask = kw.get(Lexicon.MASK, None)
        mask = [None] if mask is None else batch_extract(mask)
        func = kw.get(Lexicon.FUNC, [EnumBlendType.NORMAL])
        alpha = kw.get(Lexicon.A, [1])
        flip = kw.get(Lexicon.FLIP, [False])
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0), clip_min=0, clip_max=255)
        invert = kw.get(Lexicon.INVERT, [False])
        params = [tuple(x) for x in zip_longest_fill(pA, pB, mask, func, alpha, flip, mode, wihi, sample, matte, invert)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (pA, pB, mask, func, alpha, flip, mode, wihi, sample, matte, invert) in enumerate(params):

            if flip:
                pA, pB = pB, pA

            if pB is None:
                pB = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, (0,0,0,255), EnumImageType.BGRA)
            else:
                pB = tensor2cv(pB)

            matte = pixel_eval(matte, EnumImageType.BGRA)
            if pA is None:
                h, w = pB.shape[:2]
                pA = channel_solid(w, h, matte, chan=EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)
                pA = image_matte(pA, matte)

            if mask is None:
                mask = image_mask(pB)
            else:
                mask = tensor2cv(mask, EnumImageType.GRAYSCALE)

            if invert:
                mask = 255 - mask

            func = EnumBlendType[func]
            img = image_blend(pA, pB, mask, func, alpha)
            mode = EnumScaleMode[mode]
            if mode != EnumScaleMode.NONE:
                w, h = wihi
                sample = EnumInterpolation[sample]
                img = image_scalefit(img, w, h, mode, sample, matte)
            img = cv2tensor_full(img, matte)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))

class PixelSplitNode(JOVImageMultiple):
    NAME = "PIXEL SPLIT (JOV) 💔"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Splits images into constituent R, G and B and A channels."
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK",)
    RETURN_NAMES = (Lexicon.RI, Lexicon.GI, Lexicon.BI, Lexicon.MI)
    OUTPUT_IS_LIST = (True, True, True, True, )
    SORT = 40

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL)
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-pixel-split")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        pixel = kw.get(Lexicon.PIXEL, [None])
        pbar = comfy.utils.ProgressBar(len(pixel))
        for idx, (img,) in enumerate(pixel):
            img = tensor2cv(img)
            img = image_mask_add(img)
            img = [cv2tensor(image_grayscale(x)) for x in image_split(img)]
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))

class PixelMergeNode(JOVImageMultiple):
    NAME = "PIXEL MERGE (JOV) 🫂"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Combine 3 or 4 inputs into a single image."
    SORT = 45

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
        }}
        d = deep_merge_dict(IT_REQUIRED, IT_RGBA_IMAGE, d)
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-pixel-merge")

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        R = kw.get(Lexicon.R, [None])
        G = kw.get(Lexicon.G, [None])
        B = kw.get(Lexicon.B, [None])
        A = kw.get(Lexicon.A, [None])
        if len(R)+len(B)+len(G)+len(A) == 0:
            img = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 0, EnumImageType.BGRA)
            return list(cv2tensor_full(img, matte))
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0), clip_min=0, clip_max=255)
        params = [tuple(x) for x in zip_longest_fill(R, G, B, A, matte)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (r, g, b, a, matte) in enumerate(params):
            r = tensor2cv(r, chan=EnumImageType.GRAYSCALE)
            g = tensor2cv(g, chan=EnumImageType.GRAYSCALE)
            b = tensor2cv(b, chan=EnumImageType.GRAYSCALE)
            mask = tensor2cv(a, chan=EnumImageType.GRAYSCALE)
            img = channel_merge([b, g, r, mask])
            # logger.debug(img.shape)
            img = cv2tensor_full(img, matte)
            images.append(img)
            pbar.update_absolute(idx)
        data = list(zip(*images))
        return data

class StackNode(JOVImageMultiple):
    NAME = "STACK (JOV) ➕"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Union multiple images horizontal, vertical or in a grid."
    OUTPUT_IS_LIST = (False, False, False,)
    SORT = 55

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.AXIS: (EnumOrientation._member_names_, {"default": EnumOrientation.GRID.name}),
                Lexicon.STEP: ("INT", {"min": 0, "step": 1, "default": 0}),
            }}
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL, d, IT_WHMODE)
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-stack")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
        axis = kw.get(Lexicon.AXIS, [EnumOrientation.GRID])[0]
        axis = EnumOrientation[axis]
        stride = kw.get(Lexicon.STEP, [0])[0]
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])[0]
        mode = EnumScaleMode[mode]
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), clip_min=1)[0]
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])[0]
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)[0]
        matte = pixel_eval(matte)
        images = [tensor2cv(img) for img in pA]
        print(len(images))
        img = image_stack(images, axis, stride, matte)
        w, h = wihi
        if mode != EnumScaleMode.NONE:
            sample = EnumInterpolation[sample]
            img = image_scalefit(img, w, h, mode, sample)
        return cv2tensor_full(img, matte)

class EnumCropMode(Enum):
    CENTER = 20
    XY = 0
    FREE = 10

class CropNode(JOVImageMultiple):
    NAME = "CROP (JOV) ✂️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Clip away sections of an image and backfill with optional color matte."
    SORT = 5

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.FUNC: (EnumCropMode._member_names_, {"default": EnumCropMode.CENTER.name}),
            Lexicon.XY: ("VEC2", {"default": (0, 0), "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.WH: ("VEC2", {"default": (512, 512), "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.TLTR: ("VEC4", {"default": (0, 0, 0, 1), "step": 0.01, "precision": 5, "round": 0.000001, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
            Lexicon.BLBR: ("VEC4", {"default": (1, 0, 1, 1), "step": 0.01, "precision": 5, "round": 0.000001, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
        }}
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL, d, IT_RGB)
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-crop")

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
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, func, xy, wihi, tltr, blbr, color) in enumerate(params):
            img = tensor2cv(img)
            width, height = wihi
            func = EnumCropMode[func]
            if func == EnumCropMode.FREE:
                y1, x1, y2, x2 = tltr
                y4, x4, y3, x3 = blbr
                points = [(x1 * width, y1 * height), (x2 * width, y2 * height),
                          (x3 * width, y3 * height), (x4 * width, y4 * height)]
                img = image_crop_polygonal(img, points)
            elif func == EnumCropMode.XY:
                img = image_crop(img, width, height, xy)
            else:
                img = image_crop_center(img, width, height)

            img = cv2tensor_full(img, color)
            images.append(img)
            pbar.update_absolute(idx)
        return list(zip(*images))

class ColorTheoryNode(JOVImageMultiple):
    NAME = "COLOR THEORY (JOV) 🛞"
    CATEGORY = JOV_CATEGORY
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
        d = deep_merge_dict(IT_REQUIRED, IT_PIXEL, d, IT_INVERT)
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-color-theory")

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        scheme = kw.get(Lexicon.SCHEME, [EnumColorTheory.COMPLIMENTARY])
        user = parse_number(Lexicon.VALUE, kw, EnumTupleType.INT, [0], clip_min=-180, clip_max=180)
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], clip_min=0, clip_max=1)
        params = [tuple(x) for x in zip_longest_fill(pixels, scheme, user, i)]
        images = []
        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (img, s, user, i) in enumerate(params):
            img = tensor2cv(img)
            stuff = color_theory(img, user, EnumColorTheory[s])
            if i != 0:
                stuff = (image_invert(s, i) for s in stuff)
            images.append([cv2tensor(a) for a in stuff])
            pbar.update_absolute(idx)
        return list(zip(*images))
