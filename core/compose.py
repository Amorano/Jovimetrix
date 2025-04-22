""" Jovimetrix - Composition """

import cv2
import numpy as np

from comfy.utils import ProgressBar

from cozy_comfyui import \
    IMAGE_SIZE_MIN, \
    InputType, RGBAMaskType, EnumConvertType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyBaseNode, CozyImageNode

from cozy_comfyui.image import \
    EnumImageType

from cozy_comfyui.image.convert import \
    image_mask, image_matte, image_convert, tensor_to_cv, \
    cv_to_tensor, cv_to_tensor_full

from cozy_comfyui.image.misc import \
    image_minmax, image_stack

from .. import \
    Lexicon

from ..sup.image.color import \
    pixel_eval

from ..sup.image.adjust import \
    EnumScaleMode, EnumInterpolation, EnumThreshold, EnumThresholdAdapt, \
    image_contrast, image_equalize, image_filter, image_gamma,  \
    image_hsv, image_invert, image_pixelate, image_posterize, \
    image_quantize, image_scalefit, image_sharpen, image_swap_channels, \
    image_threshold, morph_edge_detect, morph_emboss

from ..sup.image.channel import \
    EnumPixelSwizzle, \
    channel_merge, channel_solid

from ..sup.image.compose import \
    EnumAdjustOP, EnumBlendType, \
    image_levels, image_split, image_blend

JOV_CATEGORY = "COMPOSE"

# ==============================================================================
# === CLASS ===
# ==============================================================================

class AdjustNode(CozyImageNode):
    NAME = "ADJUST (JOV) 🕸️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Enhance and modify images with various effects such as blurring, sharpening, color tweaks, and edge detection. Customize parameters like radius, value, and contrast, and use masks for selective effects.

Advanced options include pixelation, quantization, and morphological operations like dilation and erosion. Handle transparency effortlessly to ensure seamless blending of effects. This node is ideal for simple adjustments and complex image transformations.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {}),
                Lexicon.MASK: (COZY_TYPE_IMAGE, {}),
                Lexicon.FUNC: (EnumAdjustOP._member_names_, {"default": EnumAdjustOP.BLUR.name,
                                                            "tooltip":"Type of adjustment (e.g., blur, sharpen, invert)"}),
                Lexicon.RADIUS: ("INT", {"default": 3, "min": 3}),
                "VAL": ("FLOAT", {"default": 1, "min": 0, "step": 0.01}),
                "LoHi": ("VEC2", {"default": (0, 1),
                                        "mij": 0, "maj": 1, "step": 0.01, "label": ["Low", "HI"]}),
                "LMH": ("VEC3", {"default": (0, 0.5, 1),
                                        "mij": 0, "maj": 1, "step": 0.01, "label": ["Low", "MID", "HI"]}),
                "HSV": ("VEC3",{"default": (0, 1, 1),
                                    "mij": 0, "maj": 1, "step": 0.01,  "label": [Lexicon.H, Lexicon.S, Lexicon.V]}),
                Lexicon.CONTRAST: ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                Lexicon.GAMMA: ("FLOAT", {"default": 1, "min": 0.00001, "max": 1, "step": 0.01}),
                "MATTE": ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.IMAGE, None)
        op = parse_param(kw, Lexicon.FUNC, EnumAdjustOP, EnumAdjustOP.BLUR.name)
        radius = parse_param(kw, Lexicon.RADIUS, EnumConvertType.INT, 3, 3)
        val = parse_param(kw, "VAL", EnumConvertType.FLOAT, 0, 0)
        lohi = parse_param(kw, "LoHi", EnumConvertType.VEC2, [(0, 1)], 0, 1)
        lmh = parse_param(kw, "LMH", EnumConvertType.VEC3, [(0, 0.5, 1)], 0, 1)
        hsv = parse_param(kw, "HSV", EnumConvertType.VEC3, [(0, 1, 1)], 0, 1)
        contrast = parse_param(kw, Lexicon.CONTRAST, EnumConvertType.FLOAT, 1, 0, 1)
        gamma = parse_param(kw, Lexicon.GAMMA, EnumConvertType.FLOAT, 1, 0, 1)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, mask, op, radius, val, lohi,
                                        lmh, hsv, contrast, gamma, matte, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mask, op, radius, val, lohi, lmh, hsv, contrast, gamma, matte, invert) in enumerate(params):
            pA = tensor_to_cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGR)
            img_new = image_convert(pA, 3)

            match op:
                case EnumAdjustOP.INVERT:
                    img_new = image_invert(img_new, val)

                case EnumAdjustOP.LEVELS:
                    l, m, h = lmh
                    img_new = image_levels(img_new, l, h, m, gamma)

                case EnumAdjustOP.HSV:
                    h, s, v = hsv
                    img_new = image_hsv(img_new, h, s, v)
                    if contrast != 0:
                        img_new = image_contrast(img_new, 1 - contrast)

                    if gamma != 0:
                        img_new = image_gamma(img_new, gamma)

                case EnumAdjustOP.FIND_EDGES:
                    lo, hi = lohi
                    img_new = morph_edge_detect(img_new, low=lo, high=hi)

                case EnumAdjustOP.BLUR:
                    img_new = cv2.blur(img_new, (radius, radius))

                case EnumAdjustOP.STACK_BLUR:
                    r = min(radius, 1399)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.stackBlur(img_new, (r, r))

                case EnumAdjustOP.GAUSSIAN_BLUR:
                    r = min(radius, 999)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.GaussianBlur(img_new, (r, r), sigmaX=val)

                case EnumAdjustOP.MEDIAN_BLUR:
                    r = min(radius, 357)
                    if r % 2 == 0:
                        r += 1
                    img_new = cv2.medianBlur(img_new, r)

                case EnumAdjustOP.SHARPEN:
                    r = min(radius, 511)
                    if r % 2 == 0:
                        r += 1
                    img_new = image_sharpen(img_new, kernel_size=r, amount=val)

                case EnumAdjustOP.EMBOSS:
                    img_new = morph_emboss(img_new, val, radius)

                case EnumAdjustOP.EQUALIZE:
                    img_new = image_equalize(img_new)

                case EnumAdjustOP.PIXELATE:
                    img_new = image_pixelate(img_new, val / 255.)

                case EnumAdjustOP.QUANTIZE:
                    img_new = image_quantize(img_new, int(val))

                case EnumAdjustOP.POSTERIZE:
                    img_new = image_posterize(img_new, int(val))

                case EnumAdjustOP.OUTLINE:
                    img_new = cv2.morphologyEx(img_new, cv2.MORPH_GRADIENT, (radius, radius))

                case EnumAdjustOP.DILATE:
                    img_new = cv2.dilate(img_new, (radius, radius), iterations=int(val))

                case EnumAdjustOP.ERODE:
                    img_new = cv2.erode(img_new, (radius, radius), iterations=int(val))

                case EnumAdjustOP.OPEN:
                    img_new = cv2.morphologyEx(img_new, cv2.MORPH_OPEN, (radius, radius), iterations=int(val))

                case EnumAdjustOP.CLOSE:
                    img_new = cv2.morphologyEx(img_new, cv2.MORPH_CLOSE, (radius, radius), iterations=int(val))

            if mask is not None:
                mask = tensor_to_cv(mask)
                if invert:
                    mask = 255 - mask

            img_new = image_blend(pA, img_new, mask)
            if pA.ndim == 3 and pA.shape[2] == 4:
                mask = image_mask(pA)
                img_new = image_convert(img_new, 4)
                img_new[:,:,3] = mask
            #    img_new = image_mask_add(mask)

            images.append(cv_to_tensor_full(img_new, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class BlendNode(CozyImageNode):
    NAME = "BLEND (JOV) ⚗️"
    CATEGORY = JOV_CATEGORY
    SORT = 10
    DESCRIPTION = """
Combine two input images using various blending modes, such as normal, screen, multiply, overlay, etc. It also supports alpha blending and masking to achieve complex compositing effects. This node is essential for creating layered compositions and adding visual richness to images.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (COZY_TYPE_IMAGE, {"tooltip": "Background Plate"}),
                Lexicon.PIXEL_B: (COZY_TYPE_IMAGE, {"tooltip": "Image to Overlay on Background Plate"}),
                Lexicon.MASK: (COZY_TYPE_IMAGE, {"tooltip": "Optional Mask to use for Alpha Blend Operation. If empty, will use the ALPHA of B"}),
                Lexicon.FUNC: (EnumBlendType._member_names_, {"default": EnumBlendType.NORMAL.name, "tooltip": "Blending Operation"}),
                Lexicon.A: ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Amount of Blending to Perform on the Selected Operation"}),
                Lexicon.FLIP: ("BOOLEAN", {"default": False}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"}),
                "MODE": (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":IMAGE_SIZE_MIN, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                "MATTE": ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        func = parse_param(kw, Lexicon.FUNC, EnumBlendType, EnumBlendType.NORMAL.name)
        alpha = parse_param(kw, Lexicon.A, EnumConvertType.FLOAT, 1, 0, 1)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, "MODE", EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], IMAGE_SIZE_MIN)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, pB, mask, func, alpha, flip, mode, wihi, sample, matte, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, mask, func, alpha, flip, mode, wihi, sample, matte, invert) in enumerate(params):
            if flip:
                pA, pB = pB, pA

            width, height = IMAGE_SIZE_MIN, IMAGE_SIZE_MIN
            if pA is None:
                if pB is None:
                    if mask is None:
                        if mode != EnumScaleMode.MATTE:
                            width, height = wihi
                    else:
                        height, width = mask.shape[:2]
                else:
                    height, width = pB.shape[:2]
            else:
                height, width = pA.shape[:2]

            if pA is None:
                pA = channel_solid(width, height, matte, chan=EnumImageType.BGRA)
            else:
                pA = tensor_to_cv(pA)
                matted = pixel_eval(matte, EnumImageType.BGRA)
                pA = image_matte(pA, matted)

            if pB is None:
                pB = channel_solid(width, height, matte, chan=EnumImageType.BGRA)
            else:
                pB = tensor_to_cv(pB)

            if mask is not None:
                mask = tensor_to_cv(mask)
                # mask = image_grayscale(mask)
                if invert:
                    mask = 255 - mask

            img = image_blend(pA, pB, mask, func, alpha)

            if mode != EnumScaleMode.MATTE:
                # or mode != EnumScaleMode.RESIZE_MATTE:
                width, height = wihi
                img = image_scalefit(img, width, height, mode, sample)

            img = cv_to_tensor_full(img, matte)
            images.append(img)
            pbar.update_absolute(idx)
        return image_stack(images)

class FilterMaskNode(CozyImageNode):
    NAME = "FILTER MASK (JOV) 🤿"
    CATEGORY = JOV_CATEGORY
    SORT = 700
    DESCRIPTION = """
Create masks based on specific color ranges within an image. Specify the color range using start and end values and an optional fuzziness factor to adjust the range. This node allows for precise color-based mask creation, ideal for tasks like object isolation, background removal, or targeted color adjustments.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (COZY_TYPE_IMAGE, {}),
                "START": ("VEC3INT", {"default": (128, 128, 128), "rgb": True}),
                Lexicon.BOOLEAN: ("BOOLEAN", {"default": False, "tooltip": "use an end point (start->end) when calculating the filter range"}),
                "END": ("VEC3INT", {"default": (128, 128, 128), "rgb": True}),
                Lexicon.FLOAT: ("VEC3", {"default": (0.5,0.5,0.5), "mij":0, "maj":1, "step": 0.01, "tooltip": "the fuzziness use to extend the start and end range(s)"}),
                "MATTE": ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        start = parse_param(kw, "START", EnumConvertType.VEC3INT, [(128,128,128)], 0, 255)
        use_range = parse_param(kw, Lexicon.BOOLEAN, EnumConvertType.VEC3, [(0,0,0)], 0, 255)
        end = parse_param(kw, "END", EnumConvertType.VEC3INT, [(128,128,128)], 0, 255)
        fuzz = parse_param(kw, Lexicon.FLOAT, EnumConvertType.VEC3, [(0.5,0.5,0.5)], 0, 1)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, start, use_range, end, fuzz, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, start, use_range, end, fuzz, matte) in enumerate(params):
            img = np.zeros((IMAGE_SIZE_MIN, IMAGE_SIZE_MIN, 3), dtype=np.uint8) if pA is None else tensor_to_cv(pA)

            img, mask = image_filter(img, start, end, fuzz, use_range)
            if img.shape[2] == 3:
                alpha_channel = np.zeros((img.shape[0], img.shape[1], 1), dtype=img.dtype)
                img = np.concatenate((img, alpha_channel), axis=2)
            img[..., 3] = mask[:,:]
            images.append(cv_to_tensor_full(img, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class PixelMergeNode(CozyImageNode):
    NAME = "PIXEL MERGE (JOV) 🫂"
    CATEGORY = JOV_CATEGORY
    SORT = 45
    DESCRIPTION = """
Combines individual color channels (red, green, blue) along with an optional mask channel to create a composite image. This node is useful for merging separate color components into a single image for visualization or further processing.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {}),
                Lexicon.R: (COZY_TYPE_IMAGE, {}),
                Lexicon.G: (COZY_TYPE_IMAGE, {}),
                Lexicon.B: (COZY_TYPE_IMAGE, {}),
                Lexicon.A: (COZY_TYPE_IMAGE, {}),
                "MODE": (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":IMAGE_SIZE_MIN, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                "MATTE": ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
                Lexicon.FLIP: ("VEC4", {"mij":0, "maj":1, "step": 0.01, "tooltip": "Invert specific input prior to merging. R, G, B, A."}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the final merged output"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        rgba = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        R = parse_param(kw, Lexicon.R, EnumConvertType.MASK, None)
        G = parse_param(kw, Lexicon.G, EnumConvertType.MASK, None)
        B = parse_param(kw, Lexicon.B, EnumConvertType.MASK, None)
        A = parse_param(kw, Lexicon.A, EnumConvertType.MASK, None)
        mode = parse_param(kw, "MODE", EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], IMAGE_SIZE_MIN)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.VEC4, [(0, 0, 0, 0)], 0., 1.)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(rgba, R, G, B, A, mode, wihi, sample, matte, flip, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (rgba, r, g, b, a, mode, wihi, sample, matte, flip, invert) in enumerate(params):
            replace = r, g, b, a
            if rgba is not None:
                rgba = tensor_to_cv(rgba)
                rgba = image_convert(rgba, 4)
                rgba = image_split(rgba)
                img = [tensor_to_cv(replace[i]) if replace[i] is not None else x for i, x in enumerate(rgba)]
            else:
                img = [tensor_to_cv(x) if x is not None else x for x in replace]

            _, _, w_max, h_max = image_minmax(img)
            for i, x in enumerate(img):
                if x is None:
                    x = np.full((h_max, w_max), matte[i], dtype=np.uint8)
                else:
                    x = image_scalefit(x, w_max, h_max, EnumScaleMode.ASPECT)
                if flip[i] > 0:
                    x = image_invert(x, flip[i])
                img[i] = x

            img = channel_merge(img)
            img = image_invert(img, 1)

            if mode != EnumScaleMode.MATTE:
                w, h = wihi
                img = image_scalefit(img, w, h, mode, sample)

            if invert == True:
                img = image_invert(img, 1)

            images.append(cv_to_tensor_full(img, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class PixelSplitNode(CozyBaseNode):
    NAME = "PIXEL SPLIT (JOV) 💔"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK",)
    RETURN_NAMES = (Lexicon.RI, Lexicon.GI, Lexicon.BI, Lexicon.MI)
    OUTPUT_TOOLTIPS = (
        "Single channel output of Red Channel.",
        "Single channel output of Green Channel",
        "Single channel output of Blue Channel",
        "Single channel output of Alpha Channel"
    )
    SORT = 40
    DESCRIPTION = """
Takes an input image and splits it into its individual color channels (red, green, blue), along with a mask channel. This node is useful for separating different color components of an image for further processing or analysis.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        images = []
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        pbar = ProgressBar(len(pA))
        for idx, pA in enumerate(pA):
            pA = channel_solid(chan=EnumImageType.BGRA) if pA is None else tensor_to_cv(pA)
            images.append([cv_to_tensor(x, True) for x in image_split(pA)])
            pbar.update_absolute(idx)
        return image_stack(images)

class PixelSwapNode(CozyImageNode):
    NAME = "PIXEL SWAP (JOV) 🔃"
    CATEGORY = JOV_CATEGORY
    SORT = 48
    DESCRIPTION = """
Swap pixel values between two input images based on specified channel swizzle operations. Options include pixel inputs, swap operations for red, green, blue, and alpha channels, and constant values for each channel. The swap operations allow for flexible pixel manipulation by determining the source of each channel in the output image, whether it be from the first image, the second image, or a constant value.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL_A: (COZY_TYPE_IMAGE, {}),
                Lexicon.PIXEL_B: (COZY_TYPE_IMAGE, {}),
                "SWAP_R": (EnumPixelSwizzle._member_names_, {
                    "default": EnumPixelSwizzle.RED_A.name,
                    "tooltip": "Replace input Red channel with target channel or constant"}),
                "SWAP_G": (EnumPixelSwizzle._member_names_, {
                    "default": EnumPixelSwizzle.GREEN_A.name,
                    "tooltip": "Replace input Green channel with target channel or constant"}),
                "SWAP_B": (EnumPixelSwizzle._member_names_, {
                    "default": EnumPixelSwizzle.BLUE_A.name,
                    "tooltip": "Replace input Blue channel with target channel or constant"}),
                "SWAP_A": (EnumPixelSwizzle._member_names_, {
                    "default": EnumPixelSwizzle.ALPHA_A.name,
                    "tooltip": "Replace input Alpha channel with target channel or constant"}),
                "MATTE": ("VEC4INT", {
                    "default": (0, 0, 0, 255), "rgb": True})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL_A, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.PIXEL_B, EnumConvertType.IMAGE, None)
        swap_r = parse_param(kw, Lexicon.SWAP_R, EnumPixelSwizzle, EnumPixelSwizzle.RED_A.name)
        swap_g = parse_param(kw, Lexicon.SWAP_G, EnumPixelSwizzle, EnumPixelSwizzle.GREEN_A.name)
        swap_b = parse_param(kw, Lexicon.SWAP_B, EnumPixelSwizzle, EnumPixelSwizzle.BLUE_A.name)
        swap_a = parse_param(kw, Lexicon.SWAP_A, EnumPixelSwizzle, EnumPixelSwizzle.ALPHA_A.name)
        matte = parse_param(kw, "MATTE", EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)
        params = list(zip_longest_fill(pA, pB, swap_r, swap_g, swap_b, swap_a, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, swap_r, swap_g, swap_b, swap_a, matte) in enumerate(params):
            if pA is None:
                if pB is None:
                    out = channel_solid(chan=EnumImageType.BGRA)
                    images.append(cv_to_tensor_full(out))
                    pbar.update_absolute(idx)
                    continue

                h, w = pB.shape[:2]
                pA = channel_solid(w, h, chan=EnumImageType.BGRA)
            else:
                h, w = pA.shape[:2]
                pA = tensor_to_cv(pA)
                pA = image_convert(pA, 4)

            pB = tensor_to_cv(pB) if pB is not None else channel_solid(w, h, chan=EnumImageType.BGRA)
            pB = image_convert(pB, 4)
            pB = image_matte(pB, (0,0,0,0), w, h)
            pB = image_scalefit(pB, w, h, EnumScaleMode.CROP)

            out = image_swap_channels(pA, pB, (swap_r, swap_g, swap_b, swap_a), matte)

            images.append(cv_to_tensor_full(out))
            pbar.update_absolute(idx)
        return image_stack(images)

class ThresholdNode(CozyImageNode):
    NAME = "THRESHOLD (JOV) 📉"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Define a range and apply it to an image for segmentation and feature extraction. Choose from various threshold modes, such as binary and adaptive, and adjust the threshold value and block size to suit your needs. You can also invert the resulting mask if necessary. This node is versatile for a variety of image processing tasks.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {}),
                "ADAPT": ( EnumThresholdAdapt._member_names_,
                                {"default": EnumThresholdAdapt.ADAPT_NONE.name,
                                 "tooltip": "X-Men"}),
                Lexicon.FUNC: ( EnumThreshold._member_names_, {"default": EnumThreshold.BINARY.name}),
                Lexicon.THRESHOLD: ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.005}),
                Lexicon.SIZE: ("INT", {"default": 3, "min": 3, "max": 103}),
                Lexicon.INVERT: ("BOOLEAN", {"default": False, "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        mode = parse_param(kw, Lexicon.FUNC, EnumThreshold, EnumThreshold.BINARY.name)
        adapt = parse_param(kw, "ADAPT", EnumThresholdAdapt, EnumThresholdAdapt.ADAPT_NONE.name)
        threshold = parse_param(kw, Lexicon.THRESHOLD, EnumConvertType.FLOAT, 1, 0, 1)
        block = parse_param(kw, Lexicon.SIZE, EnumConvertType.INT, 3, 3)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, mode, adapt, threshold, block, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mode, adapt, th, block, invert) in enumerate(params):
            pA = tensor_to_cv(pA) if pA is not None else channel_solid(chan=EnumImageType.BGRA)
            pA = image_threshold(pA, th, mode, adapt, block)
            if invert == True:
                pA = image_invert(pA, 1)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)

'''
class HistogramNode(JOVImageSimple):
    NAME = "HISTOGRAM (JOV) 👁‍🗨"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE,)
    SORT = 40
    DESCRIPTION = """
The Histogram Node generates a histogram representation of the input image, showing the distribution of pixel intensity values across different bins. This visualization is useful for understanding the overall brightness and contrast characteristics of an image. Additionally, the node performs histogram normalization, which adjusts the pixel values to enhance the contrast of the image. Histogram normalization can be helpful for improving the visual quality of images or preparing them for further image processing tasks.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (COZY_TYPE_IMAGE, {}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.PIXEL, None), EnumConvertType.IMAGE, None)
        params = list(zip_longest_fill(pA,))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, ) in enumerate(params):
            pA = image_histogram(pA)
            pA = image_histogram_normalize(pA)
            images.append(cv_to_tensor(pA))
            pbar.update_absolute(idx)
        return image_stack(images)
'''
