"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Transformation
"""

from typing import Optional

import numpy as np
import torch

from Jovimetrix import zip_longest_fill, deep_merge_dict, tensor2cv, cv2mask, cv2tensor, \
    JOVImageInOutBaseNode, Logger, Lexicon, \
    IT_PIXELS, IT_TRS, IT_WH, IT_REQUIRED, IT_EDGE, \
    IT_WHMODE, MIN_IMAGE_SIZE, IT_TILE, IT_XY, IT_INVERT

from Jovimetrix.sup import comp
from Jovimetrix.sup.comp import geo_transform, geo_edge_wrap, geo_scalefit, geo_mirror, \
    remap_fisheye, remap_sphere, light_invert, \
    EnumScaleMode, EnumProjection, EnumMirrorMode, EnumInterpolation, IT_SAMPLE

# =============================================================================

class ComposeVec2:
    NAME = "VEC (JOV)"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "x": ("INT", {"default": 0}),
                "y": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INTEGER2",)
    FUNCTION = "op"

    def op(self, x: float, y: float) -> tuple[(int, int)]:
        return ((x, y),)

class TransformNode(JOVImageInOutBaseNode):
    NAME = "TRANSFORM (JOV) 🏝️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Translate, Rotate, Scale, Tile and Invert an input. CROP or WRAP the edges."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_TRS, IT_EDGE, IT_WH, IT_WHMODE, IT_SAMPLE)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        offset = kw.get(Lexicon.OFFSET, [None])
        angle = kw.get(Lexicon.ANGLE, [None])
        size = kw.get(Lexicon.SIZE, [None])
        wh = kw.get(Lexicon.WH, [None])
        edge = kw.get(Lexicon.EDGE, [None])
        mode = kw.get(Lexicon.MODE, [None])
        sample = kw.get(Lexicon.SAMPLE, [None])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, offset, angle, size, edge, wh, mode, sample):
            img, o, a, s, e, wh, m, rs = data
            oX, oY = o
            sX, sY = s
            w, h = wh
            if img is not None:
                img = tensor2cv(img)
                rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
                img = geo_transform(img, oX, oY, a, sX, sY, e, w, h, m, rs)
            else:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class TileNode(JOVImageInOutBaseNode):
    NAME = "TILE (JOV) 🀘"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Tile an Image with optional crop to original image size."
    SORT = 5

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_TILE, IT_WH, IT_WHMODE, IT_SAMPLE)

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        pixels = kw.get(Lexicon.PIXEL, [None])
        tile = kw.get(Lexicon.XY, [None])
        wh = kw.get(Lexicon.WH, [None])
        mode = kw.get(Lexicon.MODE, [None])
        sample = kw.get(Lexicon.SAMPLE, [None])
        masks = []
        images = []
        for img, xy, wh, m, rs in zip_longest_fill(pixels, tile, wh, mode, sample):
            w, h = wh
            x, y = xy
            w = w if w is not None else MIN_IMAGE_SIZE
            h = h if h is not None else MIN_IMAGE_SIZE
            img = tensor2cv(img)
            img = geo_edge_wrap(img, min(1, x or 1), min(1, y or 1))
            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
            img = geo_scalefit(img, w, h, m, rs)
            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class MirrorNode(JOVImageInOutBaseNode):
    NAME = "MIRROR (JOV) 🔰"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Flip an input across the X axis, the Y Axis or both, with independent centers."
    SORT = 25

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                Lexicon.MODE: (EnumMirrorMode._member_names_, {"default": EnumMirrorMode.X.name}),
            },
        }
        return deep_merge_dict(IT_PIXELS, d, IT_XY, IT_INVERT)

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pixels = kw.get(Lexicon.PIXEL, [None])
        offset = kw.get(Lexicon.X, [None])
        invert = kw.get(Lexicon.INVERT, [None])
        mode = kw.get(Lexicon.MODE, [None])
        masks = []
        images = []
        for img, xy, m, i in zip_longest_fill(pixels, offset, mode, invert):
            x, y = xy
            img = tensor2cv(img) if img is not None else np.zeros((0,0,3), dtype=np.uint8)
            if 'X' in m:
                img = geo_mirror(img, x or 0, 1, invert=i or 0)
            if 'Y' in m:
                img = geo_mirror(img, y or 0, 0, invert=i or 0)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class ProjectionNode(JOVImageInOutBaseNode):
    NAME = "PROJECTION (JOV) 🗺️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = ""
    SORT = 55

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
                Lexicon.PROJECTION: (EnumProjection._member_names_, {"default": EnumProjection.SPHERICAL.name}),
                Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:

        pixels = kw.get(Lexicon.PIXEL, [None])
        invert = kw.get(Lexicon.INVERT, [None])
        wh = kw.get(Lexicon.WH, [None])
        proj = kw.get(Lexicon.PROJECTION, [None])
        strength = kw.get(Lexicon.STRENGTH, [None])
        mode = kw.get(Lexicon.MODE, [None])
        sample = kw.get(Lexicon.SAMPLE, [None])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, proj, strength, wh, mode, invert, sample):
            img, pr, st, wh, m, i, rs = data
            w, h = wh
            w = w if w is not None else 0
            h = h if h is not None else 0
            st = st if st is not None else 1
            pr = pr if pr is not None else EnumProjection.SPHERICAL
            m = m if m is not None else EnumScaleMode.NONE
            img = tensor2cv(img) if img is not None else np.zeros((h, w, 3), dtype=np.uint8)
            match pr:
                case EnumProjection.SPHERICAL:
                    img = remap_sphere(img, st)
                case EnumProjection.FISHEYE:
                    img = remap_fisheye(img, st)

            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
            img = geo_scalefit(img, w, h, m, rs)
            if (i if i is not None else 0) != 0:
                img = light_invert(img, i)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass