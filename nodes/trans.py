"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Transformation
"""

from typing import Optional

import torch

from Jovimetrix import zip_longest_fill, deep_merge_dict, tensor2cv, cv2mask, cv2tensor, \
    JOVImageInOutBaseNode, Logger, \
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

    def run(self,
            pixels: list[torch.tensor],
            offsetX: Optional[list[float]],
            offsetY: Optional[list[float]],
            angle: Optional[list[float]],
            sizeX: Optional[list[float]],
            sizeY: Optional[list[float]],
            edge: Optional[list[str]],
            width: Optional[list[int]],
            height: Optional[list[int]],
            mode: Optional[list[str]],
            resample: Optional[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:

        offsetX = offsetX or [None]
        offsetY = offsetY or [None]
        angle = angle or [None]
        sizeX = sizeX or [None]
        sizeY = sizeY or [None]
        edge = edge or [None]
        width = width or [None]
        height = height or [None]
        mode = mode or [None]
        resample = resample or [None]

        masks = []
        images = []
        for data in zip_longest_fill(pixels, offsetX, offsetY,angle, sizeX, sizeY,
                                     edge,width, height, mode, resample):

            image, oX, oY, a, sX, sY, e, w, h, m, rs = data
            image = tensor2cv(image)
            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
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

    def run(self,
            pixels: list[torch.tensor],
            offsetX: Optional[list[float]]=None,
            offsetY: Optional[list[float]]=None,
            angle: Optional[list[float]]=None,
            sizeX: Optional[list[float]]=None,
            sizeY: Optional[list[float]]=None,
            edge: Optional[list[str]]=None) -> tuple[torch.Tensor, torch.Tensor]:

        offsetX = offsetX or [None]
        offsetY = offsetY or [None]
        angle = angle or [None]
        sizeX = sizeX or [None]
        sizeY = sizeY or [None]
        edge = edge or [None]

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

    def run(self,
            pixels: list[torch.tensor],
            tileX: Optional[list[float]]=None,
            tileY: Optional[list[float]]=None,
            width: Optional[list[int]]=None,
            height: Optional[list[int]]=None,
            mode: Optional[list[str]]=None,
            resample: Optional[list[str]]=None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        tileX = tileX or [None]
        tileY = tileY or [None]
        width = width or [None]
        height = height or [None]
        mode = mode or [None]
        resample = resample or [None]

        masks = []
        images = []
        for image, x, y, w, h, m, rs in zip_longest_fill(pixels, tileX, tileY, width,
                                                         height, mode, resample):
            w = w if w is not None else MIN_WIDTH
            h = h if h is not None else MIN_HEIGHT

            image = tensor2cv(image)
            image = comp.geo_edge_wrap(image, x, y)
            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
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
                "x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "mode": (["X", "Y", "XY", "YX"], {"default": "X"}),
            },
        }
        return deep_merge_dict(IT_PIXELS, d, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            x: list[float],
            y: list[float],
            mode: list[str],
            invert: list[float]) -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for idx, image in enumerate(pixels):
            image = tensor2cv(image)

            m = mode[min(idx, len(mode)-1)]
            i = invert[min(idx, len(invert)-1)]
            if 'X' in m:
                image = comp.geo_mirror(image, x, 1, invert=i)

            if 'Y' in m:
                image = comp.geo_mirror(image, y, 0, invert=i)

            images.append(cv2tensor(image))
            masks.append(cv2mask(image))

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
                "proj": (["SPHERICAL", "FISHEYE"], {"default": "FISHEYE"}),
                "strength": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
            }}
        return deep_merge_dict(IT_PIXELS, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

    def run(self,
            pixels: list[torch.tensor],
            proj: list[str],
            strength: list[float],
            width: list[int],
            height: list[int],
            mode: list[str],
            invert: list[float],
            resample: list[str]) -> tuple[torch.Tensor, torch.Tensor]:

        masks = []
        images = []
        for data in enumerate(pixels, proj, strength,
                              width, height, mode, invert, resample):

            image, pr, st, w, h, m, i, rs = data
            image = tensor2cv(image)
            match pr:
                case 'SPHERICAL':
                    image = comp.remap_sphere(image, st)

                case 'FISHEYE':
                    image = comp.remap_fisheye(image, st)

            rs = EnumInterpolation[rs] if rs is not None else EnumInterpolation.LANCZOS4
            image = comp.geo_scalefit(image, w, h, m, rs)

            if i != 0:
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