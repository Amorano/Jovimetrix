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
    IT_WHMODE, MIN_HEIGHT, MIN_WIDTH, IT_TILE, IT_INVERT

from Jovimetrix.sup import comp
from Jovimetrix.sup.comp import EnumInterpolation, IT_SAMPLE

# =============================================================================

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
        #offsetX = kw.get(Lexicon.X, [None])
        #offsetY = kw.get(Lexicon.Y, [None])
        angle = kw.get(Lexicon.ANGLE, [None])
        sizeX = kw.get(Lexicon.SIZE_X, [None])
        sizeY = kw.get(Lexicon.SIZE_Y, [None])
        width = kw.get(Lexicon.WIDTH, [None])
        height = kw.get(Lexicon.HEIGHT, [None])
        edge = kw.get(Lexicon.EDGE, [None])
        mode = kw.get(Lexicon.MODE, [None])
        resample = kw.get(Lexicon.RESAMPLE, [None])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, offset, angle, sizeX,  sizeY,
                                     edge, width, height, mode, resample):

            image, o, a, sX, sY, e, w, h, m, rs = data
            image = tensor2cv(image)
            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
            print(o)
            oX, oY = o
            image = comp.geo_transform(image, oX, oY, a, sX, sY, e, w, h, m, rs)
            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

        return (
            torch.stack(images),
            torch.stack(masks)
        )

class TRSNode(JOVImageInOutBaseNode):
    NAME = "TRS (JOV) 🌱"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/TRANSFORM"
    DESCRIPTION = "Translate, Rotate, Scale."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_TRS, IT_EDGE)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:

        pixels = kw.get(Lexicon.PIXEL, [None])
        offsetX = kw.get(Lexicon.X, [None])
        offsetY = kw.get(Lexicon.Y, [None])
        angle = kw.get(Lexicon.ANGLE, [None])
        sizeX = kw.get(Lexicon.SIZE_X, [None])
        sizeY = kw.get(Lexicon.SIZE_Y, [None])
        edge = kw.get(Lexicon.EDGE, [None])

        masks = []
        images = []
        for data in zip_longest_fill(pixels, offsetX, offsetY, angle, sizeX, sizeY, edge):
            image, oX, oY, a, sX, sY, e = data

            image = tensor2cv(image)
            image = comp.geo_transform(image, oX, oY, a, sX, sY, e)
            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

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
        tileX = kw.get(Lexicon.X, [None])
        tileY = kw.get(Lexicon.Y, [None])
        width = kw.get(Lexicon.WIDTH, [None])
        height = kw.get(Lexicon.HEIGHT, [None])
        mode = kw.get(Lexicon.MODE, [None])
        resample = kw.get(Lexicon.RESAMPLE, [None])
        masks = []
        images = []
        for image, x, y, w, h, m, rs in zip_longest_fill(pixels, tileX, tileY, width,
                                                         height, mode, resample):

            w = w if w else MIN_WIDTH
            h = h if h else MIN_HEIGHT
            image = tensor2cv(image)
            image = comp.geo_edge_wrap(image, x, y)
            rs = EnumInterpolation[rs] if rs else EnumInterpolation.LANCZOS4
            image = comp.geo_scalefit(image, w, h, m, rs)
            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

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
                Lexicon.X: ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.Y: ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.MODE: (["X", "Y", "XY", "YX"], {"default": "X"}),
            },
        }
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self, **kw) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        pixels = kw.get(Lexicon.PIXEL, [None])
        offsetX = kw.get(Lexicon.X, [None])
        offsetY = kw.get(Lexicon.Y, [None])
        invert = kw.get(Lexicon.INVERT, [None])
        mode = kw.get(Lexicon.MODE, [None])
        masks = []
        images = []
        for img, x, y, m, i in zip_longest_fill(pixels, offsetX, offsetY, mode, invert):

            img = tensor2cv(img) if img else np.zeros((0,0,3), dtype=np.uint8)
            if 'X' in m:
                img = comp.geo_mirror(img, x or 0, 1, invert=i or 0)
            if 'Y' in m:
                img = comp.geo_mirror(img, y or 0, 0, invert=i or 0)

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
                Lexicon.PROJECTION: (comp.EnumProjection._member_names_, {"default": comp.EnumProjection.SPHERICAL.name}),
                Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:

        pixels = kw.get(Lexicon.PIXEL, [None])
        invert = kw.get(Lexicon.INVERT, [None])
        width = kw.get(Lexicon.WIDTH, [None])
        height = kw.get(Lexicon.HEIGHT, [None])
        proj = kw.get(Lexicon.PROJECTION, [None])
        strength = kw.get(Lexicon.STRENGTH, [None])
        mode = kw.get(Lexicon.MODE, [None])
        resample = kw.get(Lexicon.RESAMPLE, [None])
        masks = []
        images = []
        for data in zip_longest_fill(pixels, proj, strength, width, height, mode, invert, resample):

            img, pr, st, w, h, m, i, rs = data
            w = w or 0
            h = h or 0
            img = tensor2cv(img) if img else np.zeros((h, w, 3), dtype=np.uint8)
            match pr:
                case 'SPHERICAL':
                    image = comp.remap_sphere(image, st)
                case 'FISHEYE':
                    image = comp.remap_fisheye(image, st)

            rs = EnumInterpolation[rs] if rs else EnumInterpolation.LANCZOS4
            image = comp.geo_scalefit(image, w, h, m, rs)
            if (i or 0) != 0:
                image = comp.light_invert(image, i)

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