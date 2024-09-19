"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

from typing import Tuple

import torch
import numpy as np
from PIL import ImageFont
from skimage.filters import gaussian
from loguru import logger

from comfy.utils import ProgressBar

from Jovimetrix import JOV_TYPE_IMAGE, JOVBaseNode, JOVImageNode, Lexicon, \
    deep_merge

from Jovimetrix.sup.util import EnumConvertType, parse_param, zip_longest_fill

from Jovimetrix.sup.image import MIN_IMAGE_SIZE, \
    EnumImageType, cv2tensor, cv2tensor_full, image_mask_add, image_matte, \
    tensor2cv, pil2cv

from Jovimetrix.sup.image.channel import channel_solid

from Jovimetrix.sup.image.compose import image_mask_binary

from Jovimetrix.sup.image.adjust import EnumEdge, EnumScaleMode, EnumInterpolation, \
    image_invert, image_rotate, image_scalefit, image_transform, image_translate

from Jovimetrix.sup.image.misc import EnumShapes, image_stereogram, shape_ellipse, \
    shape_polygon, shape_quad

from Jovimetrix.sup.text import EnumAlignment, EnumJustify, font_names, \
    text_autosize, text_draw

from Jovimetrix.sup.audio import graph_sausage

# =============================================================================

JOV_CATEGORY = "CREATE"

# =============================================================================

class ConstantNode(JOVImageNode):
    NAME = "CONSTANT (JOV) 🟪"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK)
    DESCRIPTION = """
Generate a constant image or mask of a specified size and color. It can be used to create solid color backgrounds or matte images for compositing with other visual elements. The node allows you to define the desired width and height of the output and specify the RGBA color value for the constant output. Additionally, you can input an optional image to use as a matte with the selected color.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {"tooltips":"Optional Image to Matte with Selected Color"}),
                Lexicon.RGBA_A: ("VEC4INT", {"default": (0, 0, 0, 255),
                                        "rgb": True, "tooltips": "Constant Color to Output"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512),
                                    "label": [Lexicon.W, Lexicon.H],
                                    "tooltips": "Desired Width and Height of the Color Output"}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, ...]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        matte = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        mode = parse_param(kw, Lexicon.MODE, EnumConvertType.STRING, EnumScaleMode.MATTE.name)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumConvertType.STRING, EnumInterpolation.LANCZOS4.name)
        images = []
        params = list(zip_longest_fill(pA, matte, wihi, mode, sample))
        pbar = ProgressBar(len(params))
        for idx, (pA, matte, wihi, mode, sample) in enumerate(params):
            width, height = wihi
            if pA is None:
                pA = channel_solid(width, height, matte, EnumImageType.BGRA)
                images.append(cv2tensor_full(pA))
            else:
                pA = tensor2cv(pA)
                mode = EnumScaleMode[mode]
                if mode != EnumScaleMode.MATTE:
                    sample = EnumInterpolation[sample]
                    pA = image_scalefit(pA, width, height, mode, sample)
                images.append(cv2tensor_full(pA, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class ShapeNode(JOVImageNode):
    NAME = "SHAPE GEN (JOV) ✨"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Create n-sided polygons. These shapes can be customized by adjusting parameters such as size, color, position, rotation angle, and edge blur. The node provides options to specify the shape type, the number of sides for polygons, the RGBA color value for the main shape, and the RGBA color value for the background. Additionally, you can control the width and height of the output images, the position offset, and the amount of edge blur applied to the shapes.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.SHAPE: (EnumShapes._member_names_, {"default": EnumShapes.CIRCLE.name}),
                Lexicon.SIDES: ("INT", {"default": 3, "mij": 3, "maj": 100}),
                Lexicon.RGBA_A: ("VEC4INT", {"default": (255, 255, 255, 255), "rgb": True, "tooltips": "Main Shape Color"}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True, "tooltips": "Background Color"}),
                Lexicon.WH: ("VEC2INT", {"default": (256, 256), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.XY: ("VEC2", {"default": (0, 0,), "step": 0.01, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.ANGLE: ("FLOAT", {"default": 0, "mij": -180, "maj": 180, "step": 0.01}),
                Lexicon.SIZE: ("VEC2", {"default": (1., 1.), "step": 0.01, "label": [Lexicon.X, Lexicon.Y]}),
                Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
                Lexicon.BLUR: ("FLOAT", {"default": 0, "mij": 0, "step": 0.01, "tooltips": "Edge blur amount (Gaussian blur)"}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = parse_param(kw, Lexicon.SHAPE, EnumConvertType.STRING, EnumShapes.CIRCLE.name)
        sides = parse_param(kw, Lexicon.SIDES, EnumConvertType.INT, 3, 3, 100)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.FLOAT, 0)
        edge = parse_param(kw, Lexicon.EDGE, EnumConvertType.STRING, EnumEdge.CLIP.name)
        offset = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, [(0, 0)])
        size = parse_param(kw, Lexicon.SIZE, EnumConvertType.VEC2, [(1, 1)], zero=0.001)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(256, 256)], MIN_IMAGE_SIZE)
        color = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(255, 255, 255, 255)], 0, 255)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        blur = parse_param(kw, Lexicon.BLUR, EnumConvertType.FLOAT, 0)
        params = list(zip_longest_fill(shape, sides, offset, angle, edge, size, wihi, color, matte, blur))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (shape, sides, offset, angle, edge, size, wihi, color, matte, blur) in enumerate(params):
            width, height = wihi
            sizeX, sizeY = size
            edge = EnumEdge[edge]
            shape = EnumShapes[shape]
            fill = color[:3][::-1]
            back = matte[:3]

            match shape:
                case EnumShapes.RECTANGLE:
                    pA = shape_quad(width, height, sizeX, sizeY, fill, back)

                case EnumShapes.SQUARE:
                    pA = shape_quad(width, height, sizeX, sizeX, fill, back)

                case EnumShapes.ELLIPSE:
                    pA = shape_ellipse(width, height, sizeX, sizeY, fill, back)

                case EnumShapes.CIRCLE:
                    pA = shape_ellipse(width, height, sizeX, sizeY, fill, back)

                case EnumShapes.POLYGON:
                    pA = shape_polygon(width, height, sizeX, sides, fill, back)

            pA = pil2cv(pA)
            pA = image_transform(pA, offset, angle, edge=edge)
            if blur > 0:
                # @TODO: Do blur on larger canvas to remove wrap bleed.
                pA = (gaussian(pA, sigma=blur, channel_axis=2) * 255).astype(np.uint8)

            pA = image_matte(pA, matte)
            mask = image_mask_binary(pA)
            pB = image_mask_add(pA, mask)
            matte = image_matte(pB, matte)

            images.append([cv2tensor(pB), cv2tensor(matte), cv2tensor(mask, True)])
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class StereogramNode(JOVImageNode):
    NAME = "STEREOGRAM (JOV) 📻"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Generates false perception 3D images from 2D input. Set tile divisions, noise, gamma, and shift parameters to control the stereogram's appearance.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.DEPTH: (JOV_TYPE_IMAGE, {}),
                Lexicon.TILE: ("INT", {"default": 8, "mij": 1}),
                Lexicon.NOISE: ("FLOAT", {"default": 0.33, "mij": 0, "maj": 1, "step": 0.01}),
                Lexicon.GAMMA: ("FLOAT", {"default": 0.33, "mij": 0, "maj": 1, "step": 0.01}),
                Lexicon.SHIFT: ("FLOAT", {"default": 1., "mij": -1, "maj": 1, "step": 0.01}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        depth = parse_param(kw, Lexicon.DEPTH, EnumConvertType.IMAGE, None)
        divisions = parse_param(kw, Lexicon.TILE, EnumConvertType.INT, 1, 1, 8)
        noise = parse_param(kw, Lexicon.NOISE, EnumConvertType.FLOAT, 1, 0)
        gamma = parse_param(kw, Lexicon.GAMMA, EnumConvertType.FLOAT, 1, 0)
        shift = parse_param(kw, Lexicon.SHIFT, EnumConvertType.FLOAT, 0, 1, -1)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, depth, divisions, noise, gamma, shift, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, depth, divisions, noise, gamma, shift, invert) in enumerate(params):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor2cv(pA)
            h, w = pA.shape[:2]
            depth = channel_solid(w, h, chan=EnumImageType.BGRA) if depth is None else tensor2cv(depth)
            if invert:
                depth = image_invert(depth, 1.0)
            pA = image_stereogram(pA, depth, divisions, noise, gamma, shift)
            images.append(cv2tensor_full(pA))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class StereoscopicNode(JOVBaseNode):
    NAME = "STEREOSCOPIC (JOV) 🕶️"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE, )
    DESCRIPTION = """
Simulates depth perception in images by generating stereoscopic views. It accepts an optional input image for color matte. Adjust baseline and focal length for customized depth effects.
"""
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {"tooltips":"Optional Image to Matte with Selected Color"}),
                Lexicon.INT: ("FLOAT", {"default": 0.1, "mij": 0, "maj": 1, "step": 0.01, "tooltips":"Baseline"}),
                Lexicon.FOCAL: ("FLOAT", {"default": 500, "mij": 0, "step": 0.01}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        baseline = parse_param(kw, Lexicon.INT, EnumConvertType.FLOAT, 0, 0.1, 1)
        focal_length = parse_param(kw, Lexicon.VALUE, EnumConvertType.FLOAT, 500, 0)
        images = []
        params = list(zip_longest_fill(pA, baseline, focal_length))
        pbar = ProgressBar(len(params))
        for idx, (pA, baseline, focal_length) in enumerate(params):
            pA = tensor2cv(pA) if pA is not None else channel_solid(chan=EnumImageType.GRAYSCALE)
            # Convert depth image to disparity map
            disparity_map = np.divide(1.0, pA.astype(np.float32), where=pA!=0)
            # Compute disparity values based on baseline and focal length
            disparity_map *= baseline * focal_length
            images.append(cv2tensor(pA))
            pbar.update_absolute(idx)
        return torch.cat(images, dim=0)

class TextNode(JOVImageNode):
    NAME = "TEXT GEN (JOV) 📝"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    FONTS = font_names()
    FONT_NAMES = sorted(FONTS.keys())
    DESCRIPTION = """
Generates images containing text based on parameters such as font, size, alignment, color, and position. Users can input custom text messages, select fonts from a list of available options, adjust font size, and specify the alignment and justification of the text. Additionally, the node provides options for auto-sizing text to fit within specified dimensions, controlling letter-by-letter rendering, and applying edge effects such as clipping and inversion.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.STRING: ("STRING", {"default": "jovimetrix", "multiline": True,
                                            "dynamicPrompts": False,
                                            "tooltips": "Your Message"}),
                Lexicon.FONT: (cls.FONT_NAMES, {"default": cls.FONT_NAMES[0]}),
                Lexicon.LETTER: ("BOOLEAN", {"default": False}),
                Lexicon.AUTOSIZE: ("BOOLEAN", {"default": False}),
                Lexicon.RGBA_A: ("VEC4INT", {"default": (255, 255, 255, 255), "rgb": True, "tooltips": "Color of the letters"}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
                Lexicon.COLUMNS: ("INT", {"default": 0, "mij": 0}),
                # if auto on, hide these...
                Lexicon.FONT_SIZE: ("INT", {"default": 16, "mij": 8}),
                Lexicon.ALIGN: (EnumAlignment._member_names_, {"default": EnumAlignment.CENTER.name}),
                Lexicon.JUSTIFY: (EnumJustify._member_names_, {"default": EnumJustify.CENTER.name}),
                Lexicon.MARGIN: ("INT", {"default": 0, "mij": -1024, "maj": 1024}),
                Lexicon.SPACING: ("INT", {"default": 25, "mij": -1024, "maj": 1024}),
                Lexicon.WH: ("VEC2INT", {"default": (256, 256),
                                    "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.XY: ("VEC2", {"default": (0, 0,), "mij": -1, "maj": 1, "step": 0.01,
                                      "label": [Lexicon.X, Lexicon.Y],
                                      "tooltips":"Offset the position"}),
                Lexicon.ANGLE: ("FLOAT", {"default": 0, "step": 0.01}),
                Lexicon.EDGE: (EnumEdge._member_names_, {"default": EnumEdge.CLIP.name}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltips": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        full_text = parse_param(kw, Lexicon.STRING, EnumConvertType.STRING, "jovimetrix")
        font_idx = parse_param(kw, Lexicon.FONT, EnumConvertType.STRING, self.FONT_NAMES[0])
        autosize = parse_param(kw, Lexicon.AUTOSIZE, EnumConvertType.BOOLEAN, False)
        letter = parse_param(kw, Lexicon.LETTER, EnumConvertType.BOOLEAN, False)
        color = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(255,255,255,255)], 0, 255)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0,0,0,255)], 0, 255)
        columns = parse_param(kw, Lexicon.COLUMNS, EnumConvertType.INT, 0)
        font_size = parse_param(kw, Lexicon.FONT_SIZE, EnumConvertType.INT, 1)
        align = parse_param(kw, Lexicon.ALIGN, EnumConvertType.STRING, EnumAlignment.CENTER.name)
        justify = parse_param(kw, Lexicon.JUSTIFY, EnumConvertType.STRING, EnumJustify.CENTER.name)
        margin = parse_param(kw, Lexicon.MARGIN, EnumConvertType.INT, 0)
        line_spacing = parse_param(kw, Lexicon.SPACING, EnumConvertType.INT, 25)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        pos = parse_param(kw, Lexicon.XY, EnumConvertType.VEC2, [(0, 0)], -1, 1)
        angle = parse_param(kw, Lexicon.ANGLE, EnumConvertType.INT, 0)
        edge = parse_param(kw, Lexicon.EDGE, EnumConvertType.STRING, EnumEdge.CLIP.name)
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
            align = EnumAlignment[align]
            justify = EnumJustify[justify]
            edge = EnumEdge[edge]
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
                images.append(cv2tensor_full(img, matte))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

class WaveGraphNode(JOVImageNode):
    NAME = "WAVE GRAPH (JOV) ▶ ılıılı"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
The Wave Graph node visualizes audio waveforms as bars. Adjust parameters like the number of bars, bar thickness, and colors.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.WAVE: ("AUDIO", {"default": None, "tooltips": "Audio Wave Object"}),
                Lexicon.VALUE: ("INT", {"default": 100, "mij": 32, "maj": 8192,
                                        "tooltips": "Number of Vertical bars to try to fit within the specified Width x Height"}),
                Lexicon.THICK: ("FLOAT", {"default": 0.72, "mij": 0, "maj": 1, "step": 0.01,
                                        "tooltips": "The percentage of fullness for each bar; currently scaled from the left only"}),
                Lexicon.WH: ("VEC2INT", {"default": (256, 256), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H], "tooltips": "Final output size of the wave bar graph"}),
                Lexicon.RGBA_A: ("VEC4INT", {"default": (128, 128, 0, 255), "rgb": True, "tooltips": "Bar Color"}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 128, 128, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
        wave = parse_param(kw, Lexicon.WAVE, EnumConvertType.ANY, [0])
        bars = parse_param(kw, Lexicon.VALUE, EnumConvertType.INT, 50, 1, 8192)
        thick = parse_param(kw, Lexicon.THICK, EnumConvertType.FLOAT, 0.75, 0, 1)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)
        rgb_a = parse_param(kw, Lexicon.RGBA_A, EnumConvertType.VEC4INT, [(196, 0, 196)], 0, 255)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(42, 12, 42, 255)], 0, 255)
        params = list(zip_longest_fill(wave, bars, wihi, thick, rgb_a, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (wave, bars, wihi, thick, rgb_a, matte) in enumerate(params):
            width, height = wihi
            if wave is None:
                img = channel_solid(width, height, matte, EnumImageType.BGRA)
            else:
                img = graph_sausage(wave[0], bars, width, height, thickness=thick, color_line=rgb_a, color_back=matte)
            images.append(cv2tensor_full(img))
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]
