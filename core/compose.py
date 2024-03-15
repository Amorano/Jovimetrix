"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition
"""

import torch
from enum import Enum
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOVImageMultiple, JOV_HELP_URL, WILDCARD, MIN_IMAGE_SIZE
from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import parse_number, parse_tuple, zip_longest_fill, EnumTupleType
from Jovimetrix.sup.image import batch_extract, channel_merge, \
    channel_solid, channel_swap, cv2tensor_full, \
    image_crop, image_crop_center, image_crop_polygonal, image_grayscale, \
    image_mask, image_mask_add, image_matte, image_transform, \
    image_split, pixel_eval, tensor2cv, \
    image_edge_wrap, image_scalefit, cv2tensor, \
    image_stack, image_mirror, image_blend, \
    color_theory, remap_fisheye, remap_perspective, remap_polar, \
    remap_sphere, image_invert, \
    EnumImageType, EnumColorTheory, EnumProjection, \
    EnumScaleMode, EnumInterpolation, EnumBlendType, \
    EnumEdge, EnumMirrorMode, EnumOrientation, EnumPixelSwap

# =============================================================================

JOV_CATEGORY = "JOVIMETRIX 🔺🟩🔵/COMPOSE"

class EnumCropMode(Enum):
    CENTER = 20
    XY = 0
    FREE = 10

# =============================================================================

class TransformNode(JOVImageMultiple):
    NAME = "TRANSFORM (JOV) 🏝️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Translate, Rotate, Scale, Tile, Mirror, Re-project and invert an input."
    SORT = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.XY: ("VEC2", {"default": (0, 0,), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.ANGLE: ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 0.01, "precision": 4, "round": 0.00001}),
            Lexicon.SIZE: ("VEC2", {"default": (1., 1.), "step": 0.01, "precision": 4, "round": 0.00001, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.TILE: ("VEC2", {"default": (1., 1.), "step": 0.1,  "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
            Lexicon.MIRROR: (EnumMirrorMode._member_names_, {"default": EnumMirrorMode.NONE.name}),
            Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.PROJECTION: (EnumProjection._member_names_, {"default": EnumProjection.NORMAL.name}),
            Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "step": 0.005, "precision": 4, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
            Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "step": 0.005, "precision": 4, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
            Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "precision": 4, "step": 0.005}),
            Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-transform")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
        offset = parse_tuple(Lexicon.XY, kw, EnumTupleType.FLOAT, (0., 0.,))
        angle = kw.get(Lexicon.ANGLE, [0])
        size = parse_tuple(Lexicon.SIZE, kw, EnumTupleType.FLOAT, (1., 1.,), zero=0.001)
        edge = kw.get(Lexicon.EDGE, [EnumEdge.CLIP])
        mirror = kw.get(Lexicon.MIRROR, [EnumMirrorMode.NONE])
        mirror_pivot = parse_tuple(Lexicon.PIVOT, kw, EnumTupleType.FLOAT, (0.5, 0.5,), 0, 1)
        tile_xy = parse_tuple(Lexicon.TILE, kw, EnumTupleType.FLOAT, (1., 1.), clip_min=1)
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
        pbar = ProgressBar(len(params))
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
            h, w = pA.shape[:2]
            edge = EnumEdge[edge]
            sample = EnumInterpolation[sample]
            pA = image_transform(pA, offset, angle, size, sample, edge)

            mirror = EnumMirrorMode[mirror]
            if mirror != EnumMirrorMode.NONE:
                mpx, mpy = mirror_pivot
                pA = image_mirror(pA, mirror, mpx, mpy)

            tx, ty = tile_xy
            if tx != 1. or ty != 1.:
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

            images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return list(zip(*images))

class BlendNode(JOVImageMultiple):
    NAME = "BLEND (JOV) ⚗️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Applies selected operation to 2 inputs with optional mask using a linear blend (alpha)."
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL_A: (WILDCARD, {"tooltip": "Background Plate"}),
            Lexicon.PIXEL_B: (WILDCARD, {"tooltip": "Image to Overlay on Background Plate"}),
            Lexicon.MASK: (WILDCARD, {"tooltip": "Optional Mask to use for Alpha Blend Operation. If empty, will use the ALPHA of B."}),
            Lexicon.FUNC: (EnumBlendType._member_names_, {"default": EnumBlendType.NORMAL.name, "tooltip": "Blending Operation"}),
            Lexicon.A: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Amount of Blending to Perform on the Selected Operation"}),
            Lexicon.FLIP: ("BOOLEAN", {"default": False}),
            Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"}),
            Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
        }}
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
        pbar = ProgressBar(len(params))
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
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-pixel-split")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
        pbar = ProgressBar(len(pA))
        for idx, (pA,) in enumerate(pA):
            pA = tensor2cv(pA)
            pA = image_mask_add(pA)
            pA = [cv2tensor(image_grayscale(x)) for x in image_split(pA)]
            images.append(pA)
            pbar.update_absolute(idx)
        return list(zip(*images))

class PixelMergeNode(JOVImageMultiple):
    NAME = "PIXEL MERGE (JOV) 🫂"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Combine 3 or 4 inputs into a single image."
    SORT = 45

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.R: (WILDCARD, {}),
            Lexicon.G: (WILDCARD, {}),
            Lexicon.B: (WILDCARD, {}),
            Lexicon.A: (WILDCARD, {}),
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
        }}
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
        pbar = ProgressBar(len(params))
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

class PixelSwapNode(JOVImageMultiple):
    NAME = "PIXEL SWAP (JOV) 🔃"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Swap inputs of one image with another or fill its channels with solids."
    SORT = 48

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.PIXEL_B: (WILDCARD, {}),
            Lexicon.SWAP_R: (EnumPixelSwap._member_names_, {"default": EnumPixelSwap.PASSTHRU.name}),
            Lexicon.R: ("INT", {"default": 0, "step": 1, "min": 0, "max": 255}),
            Lexicon.SWAP_G: (EnumPixelSwap._member_names_, {"default": EnumPixelSwap.PASSTHRU.name}),
            Lexicon.G: ("INT", {"default": 0, "step": 1, "min": 0, "max": 255}),
            Lexicon.SWAP_B: (EnumPixelSwap._member_names_, {"default": EnumPixelSwap.PASSTHRU.name}),
            Lexicon.B: ("INT", {"default": 0, "step": 1, "min": 0, "max": 255}),
            Lexicon.SWAP_A: (EnumPixelSwap._member_names_, {"default": EnumPixelSwap.PASSTHRU.name}),
            Lexicon.A: ("INT", {"default": 0, "step": 1, "min": 0, "max": 255})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-pixel-swap")

    def run(self, **kw)  -> tuple[torch.Tensor, torch.Tensor]:
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
        pB = kw.get(Lexicon.PIXEL_B, None)
        pB = [None] if pB is None else batch_extract(pB)
        swap_r = kw.get(Lexicon.SWAP_R, [EnumPixelSwap.PASSTHRU])
        r = kw.get(Lexicon.R, [0])
        swap_g = kw.get(Lexicon.SWAP_G, [EnumPixelSwap.PASSTHRU])
        g = kw.get(Lexicon.G, [0])
        swap_b = kw.get(Lexicon.SWAP_B, [EnumPixelSwap.PASSTHRU])
        b = kw.get(Lexicon.B, [0])
        swap_a = kw.get(Lexicon.SWAP_A, [EnumPixelSwap.PASSTHRU])
        a = kw.get(Lexicon.A, [0])
        params = [tuple(x) for x in zip_longest_fill(pA, pB, r, swap_r, g, swap_g,
                                                     b, swap_b, a, swap_a)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, r, swap_r, g, swap_g, b, swap_b, a, swap_a) in enumerate(params):
            if pA is None:
                pA = channel_solid(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, chan=EnumImageType.BGRA)
            else:
                pA = tensor2cv(pA)

            h, w = pA.shape[:2]
            if pB is None:
                pB = channel_solid(w, h, chan=EnumImageType.BGRA)
            else:
                pB = tensor2cv(pB)
                pB = image_crop_center(pB, w, h)
                pB = image_matte(pB, width=w, height=h)

            for _, swap in enumerate([(swap_b, b), (swap_g, g), (swap_r, r), (swap_a, a)]):
                swap, matte = swap
                swap = EnumPixelSwap[swap]
                if swap != EnumPixelSwap.PASSTHRU:
                    pA = channel_swap(pA, swap, pB, matte)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        data = list(zip(*images))
        return data

class StackNode(JOVImageMultiple):
    NAME = "STACK (JOV) ➕"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Union multiple images horizontal, vertical or in a grid."
    OUTPUT_IS_LIST = (False, False, False,)
    SORT = 75

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.AXIS: (EnumOrientation._member_names_, {"default": EnumOrientation.GRID.name}),
            Lexicon.STEP: ("INT", {"min": 1, "step": 1, "default": 1}),
            Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
            Lexicon.WH: ("VEC2", {"default": (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
            Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-stack")

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        idx = 1
        while 1:
            who = f"{Lexicon.PIXEL}_{idx}"
            if (val := kw.get(who, None)) is None:
                break
            images.extend([None] if val is None else batch_extract(val))
            idx += 1

        if len(images) == 0:
            logger.warning("no images to stack")
            return

        axis = kw.get(Lexicon.AXIS, [EnumOrientation.GRID])[0]
        axis = EnumOrientation[axis]
        stride = kw.get(Lexicon.STEP, [1])[0]
        mode = kw.get(Lexicon.MODE, [EnumScaleMode.NONE])[0]
        mode = EnumScaleMode[mode]
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), clip_min=1)[0]
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])[0]
        matte = parse_tuple(Lexicon.MATTE, kw, default=(0, 0, 0, 255), clip_min=0, clip_max=255)[0]
        matte = pixel_eval(matte, EnumImageType.BGRA)
        images = [tensor2cv(img) for img in images if img is not None]
        img = image_stack(images, axis, stride, matte)
        w, h = wihi
        if mode != EnumScaleMode.NONE:
            sample = EnumInterpolation[sample]
            img = image_scalefit(img, w, h, mode, sample)
        return cv2tensor_full(img, matte)

class CropNode(JOVImageMultiple):
    NAME = "CROP (JOV) ✂️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Clip away sections of an image and backfill with optional color matte."
    SORT = 5

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.FUNC: (EnumCropMode._member_names_, {"default": EnumCropMode.CENTER.name}),
            Lexicon.XY: ("VEC2", {"default": (0, 0), "label": [Lexicon.X, Lexicon.Y]}),
            Lexicon.WH: ("VEC2", {"default": (512, 512), "step": 1, "label": [Lexicon.W, Lexicon.H]}),
            Lexicon.TLTR: ("VEC4", {"default": (0, 0, 0, 1), "step": 0.01, "precision": 5, "round": 0.000001, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
            Lexicon.BLBR: ("VEC4", {"default": (1, 0, 1, 1), "step": 0.01, "precision": 5, "round": 0.000001, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
            Lexicon.RGB: ("VEC3", {"default": (0, 0, 0),  "step": 1, "label": [Lexicon.R, Lexicon.G, Lexicon.B], "rgb": True})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-crop")

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pA = kw.get(Lexicon.PIXEL, None)
        pA = [None] if pA is None else batch_extract(pA)
        func = kw.get(Lexicon.FUNC, [EnumCropMode.CENTER])
        # if less than 1 then use as scalar, over 1 = int(size)
        xy = parse_tuple(Lexicon.XY, kw, EnumTupleType.FLOAT, (0, 0,), 1)
        wihi = parse_tuple(Lexicon.WH, kw, default=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE,), clip_min=1)
        tltr = parse_tuple(Lexicon.TLTR, kw, EnumTupleType.FLOAT, (0, 0, 0, 1,), 0, 1)
        blbr = parse_tuple(Lexicon.BLBR, kw, EnumTupleType.FLOAT, (1, 0, 1, 1,), 0, 1)
        color = parse_tuple(Lexicon.RGB, kw, default=(0, 0, 0,), clip_min=0, clip_max=255)
        params = [tuple(x) for x in zip_longest_fill(pA, func, xy, wihi, tltr, blbr, color)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, func, xy, wihi, tltr, blbr, color) in enumerate(params):
            width, height = wihi
            if pA is not None:
                pA = tensor2cv(pA)
            else:
                pA = channel_solid(width, height)
            func = EnumCropMode[func]
            if func == EnumCropMode.FREE:
                y1, x1, y2, x2 = tltr
                y4, x4, y3, x3 = blbr
                points = [(x1 * width, y1 * height), (x2 * width, y2 * height),
                          (x3 * width, y3 * height), (x4 * width, y4 * height)]
                pA = image_crop_polygonal(pA, points)
            elif func == EnumCropMode.XY:
                pA = image_crop(pA, width, height, xy)
            else:
                pA = image_crop_center(pA, width, height)
            images.append(cv2tensor_full(pA, color))
            pbar.update_absolute(idx)
        return list(zip(*images))

class ColorTheoryNode(JOVImageMultiple):
    NAME = "COLOR THEORY (JOV) 🛞"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = "Generate Complimentary, Triadic and Tetradic color sets."
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (Lexicon.C1, Lexicon.C2, Lexicon.C3, Lexicon.C4, Lexicon.C5)
    OUTPUT_IS_LIST = (True, True, True, True, True)
    SORT = 100

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {
        "required": {},
        "optional": {
            Lexicon.PIXEL: (WILDCARD, {}),
            Lexicon.SCHEME: (EnumColorTheory._member_names_, {"default": EnumColorTheory.COMPLIMENTARY.name}),
            Lexicon.VALUE: ("INT", {"default": 45, "min": -90, "max": 90, "step": 1, "tooltip": "Custom angle of seperation to use when calculating colors"}),
            Lexicon.INVERT: ("BOOLEAN", {"default": False})
        }}
        return Lexicon._parse(d, JOV_HELP_URL + "/COMPOSE#-color-theory")

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pA = kw.get(Lexicon.PIXEL_A, None)
        pA = [None] if pA is None else batch_extract(pA)
        scheme = kw.get(Lexicon.SCHEME, [EnumColorTheory.COMPLIMENTARY])
        user = parse_number(Lexicon.VALUE, kw, EnumTupleType.INT, [0], clip_min=-180, clip_max=180)
        invert = kw.get(Lexicon.INVERT, [False])
        params = [tuple(x) for x in zip_longest_fill(pA, scheme, user, invert)]
        images = []
        pbar = ProgressBar(len(params))
        for idx, (img, s, user, invert) in enumerate(params):
            img = tensor2cv(img)
            img = color_theory(img, user, EnumColorTheory[s])
            if invert:
                img = (image_invert(s, 1) for s in img)
            images.append([cv2tensor(a) for a in img])
            pbar.update_absolute(idx)
        return list(zip(*images))
