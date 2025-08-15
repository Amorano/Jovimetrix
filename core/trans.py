""" Jovimetrix - Transform """

import sys
from enum import Enum

from comfy.utils import ProgressBar

from cozy_comfyui import \
    logger, \
    IMAGE_SIZE_MIN, \
    InputType, RGBAMaskType, EnumConvertType, \
    deep_merge, parse_param, parse_dynamic, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyImageNode, CozyBaseNode

from cozy_comfyui.image.channel import \
    channel_solid

from cozy_comfyui.image.convert import \
    tensor_to_cv, cv_to_tensor_full, cv_to_tensor, image_mask, image_mask_add

from cozy_comfyui.image.compose import \
    EnumOrientation, EnumEdge, EnumMirrorMode, EnumScaleMode, EnumInterpolation, \
    image_edge_wrap, image_mirror, image_scalefit, image_transform, \
    image_crop, image_crop_center, image_crop_polygonal, image_stacker, \
    image_flatten

from cozy_comfyui.image.misc import \
    image_stack

from cozy_comfyui.image.mapping import \
    EnumProjection, \
    remap_fisheye, remap_perspective, remap_polar, remap_sphere

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

JOV_CATEGORY = "TRANSFORM"

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumCropMode(Enum):
    CENTER = 20
    XY = 0
    FREE = 10

# ==============================================================================
# === CLASS ===
# ==============================================================================

class CropNode(CozyImageNode):
    NAME = "CROP (JOV) ✂️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Extract a portion of an input image or resize it. It supports various cropping modes, including center cropping, custom XY cropping, and free-form polygonal cropping. This node is useful for preparing image data for specific tasks or extracting regions of interest.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNCTION: (EnumCropMode._member_names_, {
                    "default": EnumCropMode.CENTER.name}),
                Lexicon.XY: ("VEC2", {
                    "default": (0, 0), "mij": 0, "maj": 1,
                    "label": ["X", "Y"]}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij": IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"]}),
                Lexicon.TLTR: ("VEC4", {
                    "default": (0, 0, 0, 1), "mij": 0, "maj": 1,
                    "label": ["TOP", "LEFT", "TOP", "RIGHT"],}),
                Lexicon.BLBR: ("VEC4", {
                    "default": (1, 0, 1, 1), "mij": 0, "maj": 1,
                    "label": ["BOTTOM", "LEFT", "BOTTOM", "RIGHT"],}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        func = parse_param(kw, Lexicon.FUNCTION, EnumCropMode, EnumCropMode.CENTER.name)
        # if less than 1 then use as scalar, over 1 = int(size)
        xy = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, (0, 0,))
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        tltr = parse_param(kw, Lexicon.TLTR, EnumConvertType.VEC4, (0, 0, 0, 1,))
        blbr = parse_param(kw, Lexicon.BLBR, EnumConvertType.VEC4, (1, 0, 1, 1,))
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        params = list(zip_longest_fill(pA, func, xy, wihi, tltr, blbr, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, func, xy, wihi, tltr, blbr, matte) in enumerate(params):
            width, height = wihi
            pA = tensor_to_cv(pA) if pA is not None else channel_solid(width, height)
            alpha = None
            if pA.ndim == 3 and pA.shape[2] == 4:
                alpha = image_mask(pA)

            if func == EnumCropMode.FREE:
                x1, y1, x2, y2 = tltr
                x4, y4, x3, y3 = blbr
                points = (x1 * width, y1 * height), (x2 * width, y2 * height), \
                    (x3 * width, y3 * height), (x4 * width, y4 * height)
                pA = image_crop_polygonal(pA, points)
                if alpha is not None:
                    alpha = image_crop_polygonal(alpha, points)
                    pA[..., 3] = alpha[..., 0][:,:]
            elif func == EnumCropMode.XY:
                pA = image_crop(pA, width, height, xy)
            else:
                pA = image_crop_center(pA, width, height)
            images.append(cv_to_tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class FlattenNode(CozyImageNode):
    NAME = "FLATTEN (JOV) ⬇️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Combine multiple input images into a single image by summing their pixel values. This operation is useful for merging multiple layers or images into one composite image, such as combining different elements of a design or merging masks. Users can specify the blending mode and interpolation method to control how the images are combined. Additionally, a matte can be applied to adjust the transparency of the final composite image.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name,}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij":1, "int": True,
                    "label": ["W", "H"]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
                Lexicon.OFFSET: ("VEC2", {
                    "default": (0, 0), "mij":0, "int": True,
                    "label": ["X", "Y"]}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        imgs = parse_dynamic(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        if imgs is None:
            logger.warning("no images to flatten")
            return ()

        # be less dumb when merging
        pA = [tensor_to_cv(i) for i in imgs]
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), 1)[0]
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)[0]
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)[0]
        offset = parse_param(kw, Lexicon.OFFSET, EnumConvertType.VEC2INT, (0, 0), 0)[0]
        w, h = wihi
        x, y = offset
        pA = image_flatten(pA, x, y, w, h, mode=mode, sample=sample)
        pA = [cv_to_tensor_full(pA, matte)]
        return image_stack(pA)

class SplitNode(CozyBaseNode):
    NAME = "SPLIT (JOV) 🎭"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("IMAGEA", "IMAGEB",)
    OUTPUT_TOOLTIPS = (
        "Left/Top image",
        "Right/Bottom image"
    )
    DESCRIPTION = """
Split an image into two or four images based on the percentages for width and height.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.VALUE: ("FLOAT", {
                    "default": 0.5, "min": 0, "max": 1, "step": 0.001
                }),
                Lexicon.FLIP: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Horizontal split (False) or Vertical split (True)"
                }),
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name,}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        percent = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 0.5, 0, 1)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        params = list(zip_longest_fill(pA, percent, flip, mode, wihi, sample, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, percent, flip, mode, wihi, sample, matte) in enumerate(params):
            w, h = wihi
            pA = channel_solid(w, h, matte) if pA is None else tensor_to_cv(pA)

            if flip:
                size = pA.shape[1]
                percent = max(1, min(size-1, int(size * percent)))
                image_a = pA[:, :percent]
                image_b = pA[:, percent:]
            else:
                size = pA.shape[0]
                percent = max(1, min(size-1, int(size * percent)))
                image_a = pA[:percent, :]
                image_b = pA[percent:, :]

            if mode != EnumScaleMode.MATTE:
                image_a = image_scalefit(image_a, w, h, mode, sample)
                image_b = image_scalefit(image_b, w, h, mode, sample)

            images.append([cv_to_tensor(img) for img in [image_a, image_b]])
            pbar.update_absolute(idx)
        return image_stack(images)

class StackNode(CozyImageNode):
    NAME = "STACK (JOV) ➕"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Merge multiple input images into a single composite image by stacking them along a specified axis.

Options include axis, stride, scaling mode, width and height, interpolation method, and matte color.

The axis parameter allows for horizontal, vertical, or grid stacking of images, while stride controls the spacing between them.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.AXIS: (EnumOrientation._member_names_, {
                    "default": EnumOrientation.GRID.name,}),
                Lexicon.STEP: ("INT", {
                    "default": 1, "min": 0,
                    "tooltip":"How many images are placed before a new row starts (stride)"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name,}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij": IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        images = parse_dynamic(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        if len(images) == 0:
            logger.warning("no images to stack")
            return

        images = [tensor_to_cv(i) for i in images]
        axis = parse_param(kw, Lexicon.AXIS, EnumOrientation, EnumOrientation.GRID.name)[0]
        stride = parse_param(kw, Lexicon.STEP, EnumConvertType.INT, 1, 0)[0]
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)[0]
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)[0]
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)[0]
        img = image_stacker(images, axis, stride) #, matte)
        if mode != EnumScaleMode.MATTE:
            w, h = wihi
            img = image_scalefit(img, w, h, mode, sample)
        rgba, rgb, mask = cv_to_tensor_full(img, matte)
        return rgba.unsqueeze(0), rgb.unsqueeze(0), mask.unsqueeze(0)

class TransformNode(CozyImageNode):
    NAME = "TRANSFORM (JOV) 🏝️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Apply various geometric transformations to images, including translation, rotation, scaling, mirroring, tiling and perspective projection. It offers extensive control over image manipulation to achieve desired visual effects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES(prompt=True, dynprompt=True)
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.MASK: (COZY_TYPE_IMAGE, {
                    "tooltip": "Override Image mask"}),
                Lexicon.XY: ("VEC2", {
                    "default": (0, 0,), "mij": -1, "maj": 1,
                    "label": ["X", "Y"]}),
                Lexicon.ANGLE: ("FLOAT", {
                    "default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.1,}),
                Lexicon.SIZE: ("VEC2", {
                    "default": (1, 1), "mij": 0.001,
                    "label": ["X", "Y"]}),
                Lexicon.TILE: ("VEC2", {
                    "default": (1, 1), "mij": 1,
                    "label": ["X", "Y"]}),
                Lexicon.EDGE: (EnumEdge._member_names_, {
                    "default": EnumEdge.CLIP.name}),
                Lexicon.MIRROR: (EnumMirrorMode._member_names_, {
                    "default": EnumMirrorMode.NONE.name}),
                Lexicon.PIVOT: ("VEC2", {
                    "default": (0.5, 0.5), "mij": 0, "maj": 1, "step": 0.01,
                    "label": ["X", "Y"]}),
                Lexicon.PROJECTION: (EnumProjection._member_names_, {
                    "default": EnumProjection.NORMAL.name}),
                Lexicon.TLTR: ("VEC4", {
                    "default": (0, 0, 1, 0), "mij": 0, "maj": 1, "step": 0.005,
                    "label": ["TOP", "LEFT", "TOP", "RIGHT"],}),
                Lexicon.BLBR: ("VEC4", {
                    "default": (0, 1, 1, 1), "mij": 0, "maj": 1, "step": 0.005,
                    "label": ["BOTTOM", "LEFT", "BOTTOM", "RIGHT"],}),
                Lexicon.STRENGTH: ("FLOAT", {
                    "default": 1, "min": 0, "max": 1, "step": 0.005}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name,}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij": IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.IMAGE, None)
        offset = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, (0, 0), -1, 1)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        size = parse_param(kw, Lexicon.SIZE, EnumConvertType.VEC2, (1, 1), 0.001)
        edge = parse_param(kw, Lexicon.EDGE, EnumEdge, EnumEdge.CLIP.name)
        mirror = parse_param(kw, Lexicon.MIRROR, EnumMirrorMode, EnumMirrorMode.NONE.name)
        mirror_pivot = parse_param(kw, Lexicon.PIVOT, EnumConvertType.VEC2, (0.5, 0.5), 0, 1)
        tile_xy = parse_param(kw, Lexicon.TILE, EnumConvertType.VEC2, (1, 1), 1)
        proj = parse_param(kw, Lexicon.PROJECTION, EnumProjection, EnumProjection.NORMAL.name)
        tltr = parse_param(kw, Lexicon.TLTR, EnumConvertType.VEC4, (0, 0, 1, 0), 0, 1)
        blbr = parse_param(kw, Lexicon.BLBR, EnumConvertType.VEC4, (0, 1, 1, 1), 0, 1)
        strength = parse_param(kw, Lexicon.STRENGTH, EnumConvertType.FLOAT, 1, 0, 1)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        params = list(zip_longest_fill(pA, mask, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, offset, angle, size, edge, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, wihi, sample, matte) in enumerate(params):
            pA = tensor_to_cv(pA) if pA is not None else channel_solid()
            if mask is None:
                mask = image_mask(pA, 255)
            else:
                mask = tensor_to_cv(mask)
            pA = image_mask_add(pA, mask)

            h, w = pA.shape[:2]
            pA = image_transform(pA, offset, angle, size, sample, edge)
            pA = image_crop_center(pA, w, h)

            if mirror != EnumMirrorMode.NONE:
                mpx, mpy = mirror_pivot
                pA = image_mirror(pA, mirror, mpx, mpy)
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample)

            tx, ty = tile_xy
            if tx != 1. or ty != 1.:
                pA = image_edge_wrap(pA, tx / 2 - 0.5, ty / 2 - 0.5)
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample)

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
                pA = image_scalefit(pA, w, h, EnumScaleMode.FIT, sample)

            if mode != EnumScaleMode.MATTE:
                w, h = wihi
                pA = image_scalefit(pA, w, h, mode, sample)

            images.append(cv_to_tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return image_stack(images)
