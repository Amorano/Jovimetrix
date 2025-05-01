""" Jovimetrix - Creation """

import torch
import numpy as np
from PIL import ImageFont
from skimage.filters import gaussian

from comfy.utils import ProgressBar

from cozy_comfyui import \
    IMAGE_SIZE_MIN, \
    InputType, EnumConvertType, RGBAMaskType, TensorType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyBaseNode, CozyImageNode

from cozy_comfyui.image import \
    EnumImageType

from cozy_comfyui.image.misc import \
    image_stack

from cozy_comfyui.image.convert import \
    image_matte, image_mask_add, image_convert, \
    pil_to_cv, cv_to_tensor, cv_to_tensor_full, tensor_to_cv

from .. import \
    Lexicon

from ..sup.image.channel import \
    channel_solid

from ..sup.image.compose import \
    EnumShapes, \
    image_blend, shape_ellipse, shape_polygon, shape_quad, image_mask_binary, \
    image_stereogram

from ..sup.image.adjust import \
    EnumEdge, EnumScaleMode, EnumInterpolation, \
    image_invert, image_rotate, image_scalefit, image_transform, image_translate

from ..sup.text import \
    EnumAlignment, EnumJustify, \
    font_names, text_autosize, text_draw

JOV_CATEGORY = "CREATE"

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ConstantNode(CozyImageNode):
    NAME = "CONSTANT (JOV) 🟪"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Generate a constant image or mask of a specified size and color. It can be used to create solid color backgrounds or matte images for compositing with other visual elements. The node allows you to define the desired width and height of the output and specify the RGBA color value for the constant output. Additionally, you can input an optional image to use as a matte with the selected color.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {
                    "tooltip":"Optional Image to Matte with Selected Color"}),
                Lexicon.MASK: (COZY_TYPE_IMAGE, {
                    "tooltip":"Override Image mask"}),
                Lexicon.RGBA_A: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,
                    "tooltip": "Constant Color to Output"}),
                "MODE": (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "int": True,
                    "label": [Lexicon.W, Lexicon.H],
                    "tooltip": "Desired Width and Height of the Color Output"}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.IMAGE, None)
        matte = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], IMAGE_SIZE_MIN)
        mode = parse_param(kw, "MODE", EnumScaleMode, EnumScaleMode.MATTE.name)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        images = []
        params = list(zip_longest_fill(pA, mask, matte, wihi, mode, sample))
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, matte, wihi, mode, sample) in enumerate(params):
            width, height = wihi
            if mask is not None:
                mask = tensor_to_cv(mask)
            if pA is None:
                pA = channel_solid(width, height, matte, EnumImageType.BGRA)
                if mask is not None:
                    pA = image_mask_add(pA, mask)
                images.append(cv_to_tensor_full(pA))
            else:
                pA = tensor_to_cv(pA)
                pA = image_convert(pA, 4)
                if mask is not None:
                    pA = image_mask_add(pA, mask)
                if mode != EnumScaleMode.MATTE:
                    pA = image_scalefit(pA, width, height, mode, sample, matte)
                images.append(cv_to_tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class ShapeNode(CozyImageNode):
    NAME = "SHAPE GEN (JOV) ✨"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Create n-sided polygons. These shapes can be customized by adjusting parameters such as size, color, position, rotation angle, and edge blur. The node provides options to specify the shape type, the number of sides for polygons, the RGBA color value for the main shape, and the RGBA color value for the background. Additionally, you can control the width and height of the output images, the position offset, and the amount of edge blur applied to the shapes.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "SHAPE": (EnumShapes._member_names_, {
                    "default": EnumShapes.CIRCLE.name}),
                "SIDES": ("INT", {
                    "default": 3, "min": 3, "max": 100}),
                Lexicon.RGBA_A: ("VEC4", {
                    "default": (255, 255, 255, 255), "rgb": True,
                    "tooltip": "Main Shape Color"}),
                "MATTE": ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,
                    "tooltip": "Background Color"}),
                Lexicon.WH: ("VEC2", {
                    "default": (256, 256), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.XY: ("VEC2", {
                    "default": (0, 0,), "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.ANGLE: ("FLOAT", {
                    "default": 0, "min": -180, "max": 180, "step": 0.01}),
                Lexicon.SIZE: ("VEC2", {
                    "default": (1., 1.), "label": [Lexicon.X, Lexicon.Y]}),
                "EDGE": (EnumEdge._member_names_, {
                    "default": EnumEdge.CLIP.name}),
                "BLUR": ("FLOAT", {
                    "default": 0, "min": 0, "step": 0.01,
                    "tooltip": "Edge blur amount (Gaussian blur)"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        shape = parse_param(kw, "SHAPE", EnumShapes, EnumShapes.CIRCLE.name)
        sides = parse_param(kw, "SIDES", EnumConvertType.INT, 3, 3, 100)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        edge = parse_param(kw, "EDGE", EnumEdge, EnumEdge.CLIP.name)
        offset = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, [(0, 0)])
        size = parse_param(kw, Lexicon.SIZE, EnumConvertType.VEC2, [(1, 1)], zero=0.001)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(256, 256)], IMAGE_SIZE_MIN)
        color = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(255, 255, 255, 255)], 0, 255)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        blur = parse_param(kw, "BLUR", EnumConvertType.FLOAT, 0)
        params = list(zip_longest_fill(shape, sides, offset, angle, edge, size, wihi, color, matte, blur))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (shape, sides, offset, angle, edge, size, wihi, color, matte, blur) in enumerate(params):
            width, height = wihi
            sizeX, sizeY = size
            fill = color[:3][::-1]

            match shape:
                case EnumShapes.RECTANGLE | EnumShapes.SQUARE:
                    rgb = shape_quad(width, height, sizeX, sizeY, fill)

                case EnumShapes.ELLIPSE | EnumShapes.CIRCLE:
                    rgb = shape_ellipse(width, height, sizeX, sizeY, fill)

                case EnumShapes.POLYGON:
                    rgb = shape_polygon(width, height, sizeX, sides, fill)

            rgb = pil_to_cv(rgb)
            rgb = image_transform(rgb, offset, angle, edge=edge)
            mask = image_mask_binary(rgb)

            if blur > 0:
                # @TODO: Do blur on larger canvas to remove wrap bleed.
                rgb = (gaussian(rgb, sigma=blur, channel_axis=2) * 255).astype(np.uint8)
                mask = (gaussian(mask, sigma=blur, channel_axis=2) * 255).astype(np.uint8)

            back = list(matte[:3]) + [255]
            canvas = np.full((height, width, 4), back, dtype=rgb.dtype)
            rgba = image_blend(canvas, rgb, mask)
            rgba = image_mask_add(rgba, mask)
            rgb = image_convert(rgba, 3)

            images.append([cv_to_tensor(rgba), cv_to_tensor(rgb), cv_to_tensor(mask, True)])
            pbar.update_absolute(idx)
        return image_stack(images)

class StereogramNode(CozyImageNode):
    NAME = "STEREOGRAM (JOV) 📻"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Generates false perception 3D images from 2D input. Set tile divisions, noise, gamma, and shift parameters to control the stereogram's appearance.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {}),
                "DEPTH": (COZY_TYPE_IMAGE, {
                    "tooltip": "Grayscale image representing a depth map"
                }),
                "TILE": ("INT", {
                    "default": 8, "min": 1}),
                "NOISE": ("FLOAT", {
                    "default": 0.33, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.GAMMA: ("FLOAT", {
                    "default": 0.33, "min": 0, "max": 1, "step": 0.01}),
                "SHIFT": ("FLOAT", {
                    "default": 1., "min": -1, "max": 1, "step": 0.01}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        depth = parse_param(kw, "DEPTH", EnumConvertType.IMAGE, None)
        divisions = parse_param(kw, "TILE", EnumConvertType.INT, 1, 1, 8)
        noise = parse_param(kw, "NOISE", EnumConvertType.FLOAT, 1, 0)
        gamma = parse_param(kw, Lexicon.GAMMA, EnumConvertType.FLOAT, 1, 0)
        shift = parse_param(kw, "SHIFT", EnumConvertType.FLOAT, 0, 1, -1)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, depth, divisions, noise, gamma, shift, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, depth, divisions, noise, gamma, shift, invert) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor_to_cv(pA)
            h, w = pA.shape[:2]
            depth = channel_solid(w, h, chan=EnumImageType.BGRA) if depth is None else tensor_to_cv(depth)
            if invert:
                depth = image_invert(depth, 1.0)
            pA = image_stereogram(pA, depth, divisions, noise, gamma, shift)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

class StereoscopicNode(CozyBaseNode):
    NAME = "STEREOSCOPIC (JOV) 🕶️"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    DESCRIPTION = """
Simulates depth perception in images by generating stereoscopic views. It accepts an optional input image for color matte. Adjust baseline and focal length for customized depth effects.
"""
    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {
                    "tooltip":"Optional Image to Matte with Selected Color"}),
                Lexicon.INT: ("FLOAT", {
                    "default": 0.1, "min": 0, "max": 1, "step": 0.01,
                    "tooltip":"Baseline"}),
                Lexicon.FOCAL: ("FLOAT", {
                    "default": 500, "min": 0, "step": 0.01}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[TensorType]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        baseline = parse_param(kw, Lexicon.INT, EnumConvertType.FLOAT, 0, 0.1, 1)
        focal_length = parse_param(kw, "VAL", EnumConvertType.FLOAT, 500, 0)
        images = []
        params = list(zip_longest_fill(pA, baseline, focal_length))
        pbar = ProgressBar(len(params))
        for idx, (pA, baseline, focal_length) in enumerate(params):
            pA = tensor_to_cv(pA) if pA is not None else channel_solid(chan=EnumImageType.GRAYSCALE)
            # Convert depth image to disparity map
            disparity_map = np.divide(1.0, pA.astype(np.float32), where=pA!=0)
            # Compute disparity values based on baseline and focal length
            disparity_map *= baseline * focal_length
            images.append(cv_to_tensor(pA))
            pbar.update_absolute(idx)
        return torch.stack(images)

class TextNode(CozyImageNode):
    NAME = "TEXT GEN (JOV) 📝"
    CATEGORY = JOV_CATEGORY
    FONTS = font_names()
    FONT_NAMES = sorted(FONTS.keys())
    DESCRIPTION = """
Generates images containing text based on parameters such as font, size, alignment, color, and position. Users can input custom text messages, select fonts from a list of available options, adjust font size, and specify the alignment and justification of the text. Additionally, the node provides options for auto-sizing text to fit within specified dimensions, controlling letter-by-letter rendering, and applying edge effects such as clipping and inversion.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.STRING: ("STRING", {
                    "default": "jovimetrix", "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "Your Message"}),
                "FONT": (cls.FONT_NAMES, {
                    "default": cls.FONT_NAMES[0]}),
                "LETTER": ("BOOLEAN", {
                    "default": False}),
                "AUTOSIZE": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Scale based on Width & Height"}),
                Lexicon.RGBA_A: ("VEC4", {
                    "default": (255, 255, 255, 255), "rgb": True,
                    "tooltip": "Color of the letters"}),
                "MATTE": ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,
                    "tooltip": "Background Color"}),
                "COLS": ("INT", {
                    "default": 0, "min": 0}),
                # if auto on, hide these...
                "SIZE": ("INT", {
                    "default": 16, "min": 8}),
                "ALIGN": (EnumAlignment._member_names_, {
                    "default": EnumAlignment.CENTER.name,
                    "tooltip": "Top, Center or Bottom alignment"}),
                "JUSTIFY": (EnumJustify._member_names_, {
                    "default": EnumJustify.CENTER.name}),
                "MARGIN": ("INT", {
                    "default": 0, "min": -1024, "max": 1024}),
                "SPACING": ("INT", {
                    "default": 0, "min": -1024, "max": 1024}),
                Lexicon.WH: ("VEC2", {
                    "default": (256, 256), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.XY: ("VEC2", {
                    "default": (0, 0,), "mij": -1, "maj": 1,
                    "label": [Lexicon.X, Lexicon.Y],
                    "tooltip":"Offset the position"}),
                Lexicon.ANGLE: ("FLOAT", {
                    "default": 0, "step": 0.01}),
                "EDGE": (EnumEdge._member_names_, {
                    "default": EnumEdge.CLIP.name}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        full_text = parse_param(kw, Lexicon.STRING, EnumConvertType.STRING, "jovimetrix")
        font_idx = parse_param(kw, "FONT", EnumConvertType.STRING, self.FONT_NAMES[0])
        autosize = parse_param(kw, "AUTOSIZE", EnumConvertType.BOOLEAN, False)
        letter = parse_param(kw, "LETTER", EnumConvertType.BOOLEAN, False)
        color = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(255,255,255,255)], 0, 255)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0,0,0,255)], 0, 255)
        columns = parse_param(kw, "COLS", EnumConvertType.INT, 0)
        font_size = parse_param(kw, "SIZE", EnumConvertType.INT, 1)
        align = parse_param(kw, "ALIGN", EnumAlignment, EnumAlignment.CENTER.name)
        justify = parse_param(kw, "JUSTIFY", EnumJustify, EnumJustify.CENTER.name)
        margin = parse_param(kw, "MARGIN", EnumConvertType.INT, 0)
        line_spacing = parse_param(kw, "SPACING", EnumConvertType.INT, 0)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], IMAGE_SIZE_MIN)
        pos = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, [(0, 0)], -1, 1)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.INT, 0)
        edge = parse_param(kw, "EDGE", EnumEdge, EnumEdge.CLIP.name)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        images = []
        params = list(zip_longest_fill(full_text, font_idx, autosize, letter, color,
                                matte, columns, font_size, align, justify, margin,
                                line_spacing, wihi, pos, angle, edge, invert))

        pbar = ProgressBar(len(params))
        for idx, (full_text, font_idx, autosize, letter, color, matte, columns,
                font_size, align, justify, margin, line_spacing, wihi, pos,
                angle, edge, invert) in enumerate(params):

            width, height = wihi
            font_name = self.FONTS[font_idx]
            full_text = str(full_text)

            if letter:
                full_text = full_text.replace('\n', '')
                if autosize:
                    _, font_size = text_autosize(full_text[0].upper(), font_name, width, height)[:2]
                    margin = 0
                    line_spacing = 0
            else:
                if autosize:
                    wm = width - margin * 2
                    hm = height - margin * 2 - line_spacing
                    columns = 0 if columns == 0 else columns * 2 + 2
                    full_text, font_size = text_autosize(full_text, font_name, wm, hm, columns)[:2]
                full_text = [full_text]
            font_size *= 2.5

            font = ImageFont.truetype(font_name, font_size)
            for ch in full_text:
                img = text_draw(ch, font, width, height, align, justify, margin, line_spacing, color)
                img = image_rotate(img, angle, edge=edge)
                img = image_translate(img, pos, edge=edge)
                if invert:
                    img = image_invert(img, 1)
                images.append(cv_to_tensor_full(img, matte))
            pbar.update_absolute(idx)
        return image_stack(images)
