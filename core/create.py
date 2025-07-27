""" Jovimetrix - Creation """

import numpy as np
from PIL import ImageFont
from skimage.filters import gaussian

from comfy.utils import ProgressBar

from cozy_comfyui import \
    IMAGE_SIZE_MIN, \
    InputType, EnumConvertType, RGBAMaskType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyImageNode

from cozy_comfyui.image import \
    EnumImageType

from cozy_comfyui.image.adjust import \
    image_invert

from cozy_comfyui.image.channel import \
    channel_solid

from cozy_comfyui.image.compose import \
    EnumEdge, EnumScaleMode, EnumInterpolation, \
    image_rotate, image_scalefit, image_transform, image_translate, image_blend

from cozy_comfyui.image.convert import \
    image_convert, pil_to_cv, cv_to_tensor, cv_to_tensor_full, tensor_to_cv, \
    image_mask, image_mask_add, image_mask_binary

from cozy_comfyui.image.misc import \
    image_stack

from cozy_comfyui.image.shape import \
    EnumShapes, \
    shape_ellipse, shape_polygon, shape_quad

from cozy_comfyui.image.text import \
    EnumAlignment, EnumJustify, \
    font_names, text_autosize, text_draw

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

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
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {
                    "tooltip":"Optional Image to Matte with Selected Color"}),
                Lexicon.MASK: (COZY_TYPE_IMAGE, {
                    "tooltip":"Override Image mask"}),
                Lexicon.COLOR: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,
                    "tooltip": "Constant Color to Output"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name,}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij": 1, "int": True,
                    "label": ["W", "H"],}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        matte = parse_param(kw, Lexicon.COLOR, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), 1)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        images = []
        params = list(zip_longest_fill(pA, mask, matte, mode, wihi, sample))
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, matte, mode, wihi, sample) in enumerate(params):
            width, height = wihi
            w, h = width, height

            if pA is None:
                pA = channel_solid(width, height, (0,0,0,255))
            else:
                pA = tensor_to_cv(pA)
                pA = image_convert(pA, 4)
                h, w = pA.shape[:2]

            if mask is None:
                mask = image_mask(pA, 0)
            else:
                mask = tensor_to_cv(mask, invert=1, chan=1)
                mask = image_scalefit(mask, w, h, matte=(0,0,0,255), mode=EnumScaleMode.FIT)

            pB = channel_solid(w, h, matte)
            pA = image_blend(pB, pA, mask)
            #mask = image_invert(mask, 1)
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
                Lexicon.SHAPE: (EnumShapes._member_names_, {
                    "default": EnumShapes.CIRCLE.name}),
                Lexicon.SIDES: ("INT", {
                    "default": 3, "min": 3, "max": 100}),
                Lexicon.COLOR: ("VEC4", {
                    "default": (255, 255, 255, 255), "rgb": True,
                    "tooltip": "Main Shape Color"}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
                Lexicon.WH: ("VEC2", {
                    "default": (256, 256), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"],}),
                Lexicon.XY: ("VEC2", {
                    "default": (0, 0,), "mij": -1, "maj": 1,
                    "label": ["X", "Y"]}),
                Lexicon.ANGLE: ("FLOAT", {
                    "default": 0, "min": -180, "max": 180, "step": 0.01,}),
                Lexicon.SIZE: ("VEC2", {
                    "default": (1, 1), "mij": 0, "maj": 1,
                    "label": ["X", "Y"]}),
                Lexicon.EDGE: (EnumEdge._member_names_, {
                    "default": EnumEdge.CLIP.name}),
                Lexicon.BLUR: ("FLOAT", {
                    "default": 0, "min": 0, "step": 0.01,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        shape = parse_param(kw, Lexicon.SHAPE, EnumShapes, EnumShapes.CIRCLE.name)
        sides = parse_param(kw, Lexicon.SIDES, EnumConvertType.INT, 3, 3)
        color = parse_param(kw, Lexicon.COLOR, EnumConvertType.VEC4INT, (255, 255, 255, 255), 0, 255)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (256, 256), IMAGE_SIZE_MIN)
        offset = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, (0, 0), -1, 1)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0, -180, 180)
        size = parse_param(kw, Lexicon.SIZE, EnumConvertType.VEC2, (1, 1), 0, 1, zero=0.001)
        edge = parse_param(kw, Lexicon.EDGE, EnumEdge, EnumEdge.CLIP.name)
        blur = parse_param(kw, Lexicon.BLUR, EnumConvertType.FLOAT, 0, 0)
        params = list(zip_longest_fill(shape, sides, color, matte, wihi, offset, angle, size, edge, blur))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (shape, sides, color, matte, wihi, offset, angle, size, edge, blur) in enumerate(params):
            width, height = wihi
            sizeX, sizeY = size
            fill = color[:3][::-1]

            match shape:
                case EnumShapes.SQUARE:
                    rgb = shape_quad(width, height, sizeX, sizeY, fill)

                case EnumShapes.CIRCLE:
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

            mask = (mask * (color[3] / 255.)).astype(np.uint8)
            back = list(matte[:3]) + [255]
            canvas = np.full((height, width, 4), back, dtype=rgb.dtype)
            rgba = image_blend(canvas, rgb, mask)
            rgba = image_mask_add(rgba, mask)
            rgb = image_convert(rgba, 3)

            images.append([cv_to_tensor(rgba), cv_to_tensor(rgb), cv_to_tensor(mask, True)])
            pbar.update_absolute(idx)
        return image_stack(images)

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
                Lexicon.FONT: (cls.FONT_NAMES, {
                    "default": cls.FONT_NAMES[0]}),
                Lexicon.LETTER: ("BOOLEAN", {
                    "default": False,}),
                Lexicon.AUTOSIZE: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Scale based on Width & Height"}),
                Lexicon.COLOR: ("VEC4", {
                    "default": (255, 255, 255, 255), "rgb": True,
                    "tooltip": "Color of the letters"}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
                Lexicon.COLUMNS: ("INT", {
                    "default": 0, "min": 0}),
                # if auto on, hide these...
                Lexicon.SIZE: ("INT", {
                    "default": 16, "min": 8}),
                Lexicon.ALIGN: (EnumAlignment._member_names_, {
                    "default": EnumAlignment.CENTER.name,}),
                Lexicon.JUSTIFY: (EnumJustify._member_names_, {
                    "default": EnumJustify.CENTER.name,}),
                Lexicon.MARGIN: ("INT", {
                    "default": 0, "min": -1024, "max": 1024,}),
                Lexicon.SPACING: ("INT", {
                    "default": 0, "min": -1024, "max": 1024}),
                Lexicon.WH: ("VEC2", {
                    "default": (256, 256), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"],}),
                Lexicon.XY: ("VEC2", {
                    "default": (0, 0,), "mij": -1, "maj": 1,
                    "label": ["X", "Y"],
                    "tooltip":"Offset the position"}),
                Lexicon.ANGLE: ("FLOAT", {
                    "default": 0, "step": 0.01,}),
                Lexicon.EDGE: (EnumEdge._member_names_, {
                    "default": EnumEdge.CLIP.name}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        full_text = parse_param(kw, Lexicon.STRING, EnumConvertType.STRING, "jovimetrix")
        font_idx = parse_param(kw, Lexicon.FONT, EnumConvertType.STRING, self.FONT_NAMES[0])
        autosize = parse_param(kw, Lexicon.AUTOSIZE, EnumConvertType.BOOLEAN, False)
        letter = parse_param(kw, Lexicon.LETTER, EnumConvertType.BOOLEAN, False)
        color = parse_param(kw, Lexicon.COLOR, EnumConvertType.VEC4INT, (255,255,255,255))
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0,0,0,255))
        columns = parse_param(kw, Lexicon.COLUMNS, EnumConvertType.INT, 0)
        font_size = parse_param(kw, Lexicon.SIZE, EnumConvertType.INT, 1)
        align = parse_param(kw, Lexicon.ALIGN, EnumAlignment, EnumAlignment.CENTER.name)
        justify = parse_param(kw, Lexicon.JUSTIFY, EnumJustify, EnumJustify.CENTER.name)
        margin = parse_param(kw, Lexicon.MARGIN, EnumConvertType.INT, 0)
        line_spacing = parse_param(kw, Lexicon.SPACING, EnumConvertType.INT, 0)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        pos = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, (0, 0))
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.INT, 0)
        edge = parse_param(kw, Lexicon.EDGE, EnumEdge, EnumEdge.CLIP.name)
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
