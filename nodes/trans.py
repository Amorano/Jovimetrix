"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Transformation
"""

import cv2
import torch
import numpy as np

from Jovimetrix import parse_tuple, parse_number, zip_longest_fill, \
    deep_merge_dict, tensor2cv, cv2mask, cv2tensor, \
    EnumTupleType, JOVImageInOutBaseNode, EnumEdge, Lexicon, \
    IT_PIXELS, IT_TRS, IT_REQUIRED, IT_WHMODE, MIN_IMAGE_SIZE, IT_INVERT

from Jovimetrix.sup.comp import geo_rotate, geo_translate, geo_crop, \
    geo_edge_wrap, geo_scalefit, geo_mirror, light_invert, \
    EnumScaleMode, EnumMirrorMode, EnumInterpolation, \
    IT_SAMPLE

from Jovimetrix.sup.mapping import remap_fisheye, remap_perspective, remap_polar, \
    remap_sphere, EnumProjection
# =============================================================================

class TransformNode(JOVImageInOutBaseNode):
    NAME = "TRANSFORM (JOV) 🏝️"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵"
    DESCRIPTION = "Translate, Rotate, Scale, Tile, Mirror, Re-project and invert an input"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
                Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
                Lexicon.MIRROR: (EnumMirrorMode._member_names_, {"default": EnumMirrorMode.NONE.name}),
                Lexicon.PIVOT: ("VEC2", {"default": (0.5, 0.5), "max": 1, "min": "0", "step": 0.005, "precision": 4, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.TILE: ("VEC2", {"default": (1, 1), "step": 0.125, "min": 1, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.PROJECTION: (EnumProjection._member_names_, {"default": EnumProjection.NORMAL.name}),
                Lexicon.TLTR: ("VEC4", {"default": (0, 0, 1, 0), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.TOP, Lexicon.LEFT, Lexicon.TOP, Lexicon.RIGHT]}),
                Lexicon.BLBR: ("VEC4", {"default": (0, 1, 1, 1), "min": 0, "max": 1, "step": 0.005, "precision": 4, "label": [Lexicon.BOTTOM, Lexicon.LEFT, Lexicon.BOTTOM, Lexicon.RIGHT]}),
                Lexicon.STRENGTH: ("FLOAT", {"default": 1, "min": 0, "precision": 4, "step": 0.005})
            }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXELS, IT_TRS, d, IT_WHMODE, IT_SAMPLE, IT_INVERT)

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
        sample = kw.get(Lexicon.SAMPLE, [EnumInterpolation.LANCZOS4])
        i = parse_number(Lexicon.INVERT, kw, EnumTupleType.FLOAT, [1], 0, 1)
        masks = []
        images = []
        for data in zip_longest_fill(pixels, offset, angle, size, edge, wihi, tile_xy, mirror, mirror_pivot, proj, strength, tltr, blbr, mode, sample, i):
            img, offset, angle, size, edge, wihi, tile_xy, mirror, mirror_pivot, pr, str, tltr, blbr, mode, rs, i = data
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
                rs = EnumInterpolation[rs]
                if sX != 1. or sY != 1.:
                    wx =  int(max(1, w * sX))
                    hx =  int(max(1, h * sY))
                    img = cv2.resize(img, (wx, hx), interpolation=rs.value)

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
                if tx > 1 or ty > 1:
                    img = geo_edge_wrap(img, tx - 1, ty - 1)
                    img = geo_scalefit(img, w, h, EnumScaleMode.FIT)

                # img = geo_crop(img)

                # RE-PROJECTION
                pr = EnumProjection[pr]
                match pr:
                    case EnumProjection.SPHERICAL:
                        img = remap_sphere(img, str)
                    case EnumProjection.FISHEYE:
                        img = remap_fisheye(img, str)
                    case EnumProjection.POLAR:
                        img = remap_polar(img)
                    case EnumProjection.PERSPECTIVE:
                        x1, y1, x2, y2 = tltr
                        x4, y4, x3, y3 = blbr
                        sw, sh = w, h
                        if mode == EnumScaleMode.NONE:
                            sh, sw = img.shape[:2]

                        x1 *= sw
                        x2 *= sw
                        x3 *= sw
                        x4 *= sw
                        y1 *= sh
                        y2 *= sh
                        y3 *= sh
                        y4 *= sh
                        img = remap_perspective(img, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

                if i != 0:
                    img = light_invert(img, i)

                #img = geo_crop(img)
                img = geo_scalefit(img, w, h, mode, rs)

            else:
                img = np.zeros((h, w, 3), dtype=np.uint8)

            images.append(cv2tensor(img))
            masks.append(cv2mask(img))

        return (
            torch.stack(images),
            torch.stack(masks)
        )
