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
    CozyImageNode

from cozy_comfyui.image.channel import \
    channel_solid

from cozy_comfyui.image.convert import \
    tensor_to_cv, cv_to_tensor_full

from cozy_comfyui.image.compose import \
    EnumOrientation, EnumEdge, EnumMirrorMode, EnumScaleMode, EnumInterpolation, \
    image_edge_wrap, image_mirror, image_scalefit, image_transform, \
    image_crop, image_crop_center, image_crop_polygonal, image_stacker, image_flatten

from cozy_comfyui.image.mask import \
    image_mask, image_mask_add

from cozy_comfyui.image.misc import \
    image_stack

from ..sup.image.mapping import \
    EnumProjection, \
    remap_fisheye, remap_perspective, remap_polar, remap_sphere

JOV_CATEGORY = "TRANSFORM"

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumCropMode(Enum):
    CENTER = 20
    XY = 0
    FREE = 10
    HEAD = 15
    BODY = 25

# ==============================================================================
# === CLASS ===
# ==============================================================================

class CropNode(CozyImageNode):
    NAME = "CROP (JOV) ✂️"
    CATEGORY = JOV_CATEGORY
    SORT = 5
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
                    "default": (0, 0), "mij": 0.5, "maj": 0.5,
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
        xy = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, (0, 0,), 0, 1)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        tltr = parse_param(kw, Lexicon.TLTR, EnumConvertType.VEC4, (0, 0, 0, 1,), 0, 1)
        blbr = parse_param(kw, Lexicon.BLBR, EnumConvertType.VEC4, (1, 0, 1, 1,), 0, 1)
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
            elif func == EnumCropMode.HEAD:
                pass
            elif func == EnumCropMode.BODY:
                pass
            else:
                pA = image_crop_center(pA, width, height)
            images.append(cv_to_tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class FlattenNode(CozyImageNode):
    NAME = "FLATTEN (JOV) ⬇️"
    CATEGORY = JOV_CATEGORY
    SORT = 500
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
        imgs = parse_dynamic(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        if imgs is None:
            logger.warning("no images to flatten")
            return ()

        # be less dumb when merging
        pA = [tensor_to_cv(i) for i in imgs]
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)

        images = []
        params = list(zip_longest_fill(mode, sample, wihi, matte))
        pbar = ProgressBar(len(params))
        for idx, (mode, sample, wihi, matte) in enumerate(params):
            current = image_flatten(pA)
            images.append(cv_to_tensor_full(current, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class StackNode(CozyImageNode):
    NAME = "STACK (JOV) ➕"
    CATEGORY = JOV_CATEGORY
    SORT = 75
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
        images = parse_dynamic(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        if len(images) == 0:
            logger.warning("no images to stack")
            return

        images = [tensor_to_cv(i) for i in images]
        axis = parse_param(kw, Lexicon.AXIS, EnumOrientation, EnumOrientation.GRID.name)[0]
        stride = parse_param(kw, Lexicon.STEP, EnumConvertType.INT, 1)[0]
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
    SORT = 0
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
                    "default": (0., 0.,), "mij": -1., "maj": 1.,
                    "label": ["X", "Y"]}),
                Lexicon.ANGLE: ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize, "step": 0.1,}),
                Lexicon.SIZE: ("VEC2", {
                    "default": (1., 1.), "mij": 0.001,
                    "label": ["X", "Y"]}),
                Lexicon.TILE: ("VEC2", {
                    "default": (1., 1.), "mij": 1.,
                    "label": ["X", "Y"]}),
                Lexicon.EDGE: (EnumEdge._member_names_, {
                    "default": EnumEdge.CLIP.name}),
                Lexicon.MIRROR: (EnumMirrorMode._member_names_, {
                    "default": EnumMirrorMode.NONE.name}),
                Lexicon.PIVOT: ("VEC2", {
                    "default": (0.5, 0.5), "step": 0.005,
                    "label": ["X", "Y"]}),
                Lexicon.PROJECTION: (EnumProjection._member_names_, {
                    "default": EnumProjection.NORMAL.name}),
                Lexicon.TLTR: ("VEC4", {
                    "default": (0., 0., 1., 0.), "mij": 0., "maj": 1., "step": 0.005,
                    "label": ["TOP", "LEFT", "TOP", "RIGHT"],}),
                Lexicon.BLBR: ("VEC4", {
                    "default": (0., 1., 1., 1.), "mij": 0., "maj": 1., "step": 0.005,
                    "label": ["BOTTOM", "LEFT", "BOTTOM", "RIGHT"],}),
                Lexicon.STRENGTH: ("FLOAT", {
                    "default": 1, "min": 0, "step": 0.005}),
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
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.IMAGE, None)
        offset = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, (0., 0.), -2.5, 2.5)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        size = parse_param(kw, Lexicon.SIZE, EnumConvertType.VEC2, (1., 1.), 0.001)
        edge = parse_param(kw, Lexicon.EDGE, EnumEdge, EnumEdge.CLIP.name)
        mirror = parse_param(kw, Lexicon.MIRROR, EnumMirrorMode, EnumMirrorMode.NONE.name)
        mirror_pivot = parse_param(kw, Lexicon.PIVOT, EnumConvertType.VEC2, (0.5, 0.5), 0, 1)
        tile_xy = parse_param(kw, Lexicon.TILE, EnumConvertType.VEC2, (1., 1.), 1)
        proj = parse_param(kw, Lexicon.PROJECTION, EnumProjection, EnumProjection.NORMAL.name)
        tltr = parse_param(kw, Lexicon.TLTR, EnumConvertType.VEC4, (0., 0., 1., 0.), 0, 1)
        blbr = parse_param(kw, Lexicon.BLBR, EnumConvertType.VEC4, (0., 1., 1., 1.), 0, 1)
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
            if mask is not None:
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
