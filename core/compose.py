""" Jovimetrix - Composition """

import numpy as np

from comfy.utils import ProgressBar

from cozy_comfyui import \
    IMAGE_SIZE_MIN, \
    InputType, RGBAMaskType, EnumConvertType, \
    deep_merge, parse_param, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, \
    CozyBaseNode, CozyImageNode

from cozy_comfyui.image import \
    EnumImageType

from cozy_comfyui.image.adjust import \
    EnumThreshold, EnumThresholdAdapt, \
    image_histogram2, image_invert, image_filter, image_threshold

from cozy_comfyui.image.channel import \
    EnumPixelSwizzle, \
    channel_merge, channel_solid, channel_swap

from cozy_comfyui.image.compose import \
    EnumBlendType, EnumScaleMode, EnumScaleInputMode, EnumInterpolation, \
    image_resize, \
    image_scalefit, image_split, image_blend, image_matte

from cozy_comfyui.image.convert import \
    image_mask, image_convert, tensor_to_cv, cv_to_tensor, cv_to_tensor_full

from cozy_comfyui.image.misc import \
    image_by_size, image_minmax, image_stack

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

JOV_CATEGORY = "COMPOSE"

# ==============================================================================
# === CLASS ===
# ==============================================================================

class BlendNode(CozyImageNode):
    NAME = "BLEND (JOV) ⚗️"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Combine two input images using various blending modes, such as normal, screen, multiply, overlay, etc. It also supports alpha blending and masking to achieve complex compositing effects. This node is essential for creating layered compositions and adding visual richness to images.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE_BACK: (COZY_TYPE_IMAGE, {}),
                Lexicon.IMAGE_FORE: (COZY_TYPE_IMAGE, {}),
                Lexicon.MASK: (COZY_TYPE_IMAGE, {
                    "tooltip": "Optional Mask for Alpha Blending. If empty, it will use the ALPHA of the FOREGROUND"}),
                Lexicon.FUNCTION: (EnumBlendType._member_names_, {
                    "default": EnumBlendType.NORMAL.name,}),
                Lexicon.ALPHA: ("FLOAT", {
                    "default": 1, "min": 0, "max": 1, "step": 0.01,}),
                Lexicon.SWAP: ("BOOLEAN", {
                    "default": False}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False, "tooltip": "Invert the mask input"}),
                Lexicon.MODE: (EnumScaleMode._member_names_, {
                    "default": EnumScaleMode.MATTE.name,}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {
                    "default": EnumInterpolation.LANCZOS4.name,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
                Lexicon.INPUT: (EnumScaleInputMode._member_names_, {
                    "default": EnumScaleInputMode.NONE.name,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        back = parse_param(kw, Lexicon.IMAGE_BACK, EnumConvertType.IMAGE, None)
        fore = parse_param(kw, Lexicon.IMAGE_FORE, EnumConvertType.IMAGE, None)
        mask = parse_param(kw, Lexicon.MASK, EnumConvertType.MASK, None)
        func = parse_param(kw, Lexicon.FUNCTION, EnumBlendType, EnumBlendType.NORMAL.name)
        alpha = parse_param(kw, Lexicon.ALPHA, EnumConvertType.FLOAT, 1)
        swap = parse_param(kw, Lexicon.SWAP, EnumConvertType.BOOLEAN, False)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        inputMode = parse_param(kw, Lexicon.INPUT, EnumScaleInputMode, EnumScaleInputMode.NONE.name)
        params = list(zip_longest_fill(back, fore, mask, func, alpha, swap, invert, mode, wihi, sample, matte, inputMode))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (back, fore, mask, func, alpha, swap, invert, mode, wihi, sample, matte, inputMode) in enumerate(params):
            if swap:
                back, fore = fore, back

            width, height = IMAGE_SIZE_MIN, IMAGE_SIZE_MIN
            if back is None:
                if fore is None:
                    if mask is None:
                        if mode != EnumScaleMode.MATTE:
                            width, height = wihi
                    else:
                        height, width = mask.shape[:2]
                else:
                    height, width = fore.shape[:2]
            else:
                height, width = back.shape[:2]

            if back is None:
                back = channel_solid(width, height, matte)
            else:
                back = tensor_to_cv(back)
                #matted = pixel_eval(matte)
                #back = image_matte(back, matted)

            if fore is None:
                clear = list(matte[:3]) + [0]
                fore = channel_solid(width, height, clear)
            else:
                fore = tensor_to_cv(fore)

            if mask is None:
                mask = image_mask(fore, 255)
            else:
                mask = tensor_to_cv(mask, 1)

            if invert:
                mask = 255 - mask

            if inputMode != EnumScaleInputMode.NONE:
                # get the min/max of back, fore; and mask?
                imgs = [back, fore]
                _, w, h = image_by_size(imgs)
                back = image_scalefit(back, w, h, inputMode, sample, matte)
                fore = image_scalefit(fore, w, h, inputMode, sample, matte)
                mask = image_scalefit(mask, w, h, inputMode, sample)

                back = image_scalefit(back, w, h, EnumScaleMode.RESIZE_MATTE, sample, matte)
                fore = image_scalefit(fore, w, h, EnumScaleMode.RESIZE_MATTE, sample, (0,0,0,255))
                mask = image_scalefit(mask, w, h, EnumScaleMode.RESIZE_MATTE, sample, (255,255,255,255))

            img = image_blend(back, fore, mask, func, alpha)
            mask = image_mask(img)

            if mode != EnumScaleMode.MATTE:
                width, height = wihi
                img = image_scalefit(img, width, height, mode, sample, matte)

            img = cv_to_tensor_full(img, matte)
            #img = [cv_to_tensor(back), cv_to_tensor(fore), cv_to_tensor(mask, True)]
            images.append(img)
            pbar.update_absolute(idx)

        return image_stack(images)

class FilterMaskNode(CozyImageNode):
    NAME = "FILTER MASK (JOV) 🤿"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Create masks based on specific color ranges within an image. Specify the color range using start and end values and an optional fuzziness factor to adjust the range. This node allows for precise color-based mask creation, ideal for tasks like object isolation, background removal, or targeted color adjustments.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.START: ("VEC3", {
                    "default": (128, 128, 128), "rgb": True}),
                Lexicon.RANGE: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use an end point (start->end) when calculating the filter range"}),
                Lexicon.END: ("VEC3", {
                    "default": (128, 128, 128), "rgb": True}),
                Lexicon.FUZZ: ("VEC3", {
                    "default": (0.5,0.5,0.5), "mij":0, "maj":1,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        start = parse_param(kw, Lexicon.START, EnumConvertType.VEC3INT, (128,128,128), 0, 255)
        use_range = parse_param(kw, Lexicon.RANGE, EnumConvertType.BOOLEAN, False)
        end = parse_param(kw, Lexicon.END, EnumConvertType.VEC3INT, (128,128,128), 0, 255)
        fuzz = parse_param(kw, Lexicon.FUZZ, EnumConvertType.VEC3, (0.5,0.5,0.5), 0, 1)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
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

class HistogramNode(CozyImageNode):
    NAME = "HISTOGRAM (JOV)"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
The Histogram Node generates a histogram representation of the input image, showing the distribution of pixel intensity values across different bins. This visualization is useful for understanding the overall brightness and contrast characteristics of an image. Additionally, the node performs histogram normalization, which adjusts the pixel values to enhance the contrast of the image. Histogram normalization can be helpful for improving the visual quality of images or preparing them for further image processing tasks.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {
                    "tooltip": "Pixel Data (RGBA, RGB or Grayscale)"}),
                Lexicon.WH: ("VEC2", {
                    "default": (512, 512), "mij":IMAGE_SIZE_MIN, "int": True,
                    "label": ["W", "H"]}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, (512, 512), IMAGE_SIZE_MIN)
        params = list(zip_longest_fill(pA, wihi))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, wihi) in enumerate(params):
            pA = tensor_to_cv(pA) if pA is not None else channel_solid()
            hist_img = image_histogram2(pA, bins=256)
            width, height = wihi
            hist_img = image_resize(hist_img, width, height, EnumInterpolation.NEAREST)
            images.append(cv_to_tensor_full(hist_img))
            pbar.update_absolute(idx)
        return image_stack(images)

class PixelMergeNode(CozyImageNode):
    NAME = "PIXEL MERGE (JOV) 🫂"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Combines individual color channels (red, green, blue) along with an optional mask channel to create a composite image.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.CHAN_RED: (COZY_TYPE_IMAGE, {}),
                Lexicon.CHAN_GREEN: (COZY_TYPE_IMAGE, {}),
                Lexicon.CHAN_BLUE: (COZY_TYPE_IMAGE, {}),
                Lexicon.CHAN_ALPHA: (COZY_TYPE_IMAGE, {}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,}),
                Lexicon.FLIP: ("VEC4", {
                    "default": (0,0,0,0), "mij":0, "maj":1, "step": 0.01,
                    "tooltip": "Invert specific input prior to merging. R, G, B, A."}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        rgba = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        R = parse_param(kw, Lexicon.CHAN_RED, EnumConvertType.MASK, None)
        G = parse_param(kw, Lexicon.CHAN_GREEN, EnumConvertType.MASK, None)
        B = parse_param(kw, Lexicon.CHAN_BLUE, EnumConvertType.MASK, None)
        A = parse_param(kw, Lexicon.CHAN_ALPHA, EnumConvertType.MASK, None)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        flip = parse_param(kw, Lexicon.FLIP, EnumConvertType.VEC4, (0, 0, 0, 0), 0, 1)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(rgba, R, G, B, A, matte, flip, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (rgba, r, g, b, a, matte, flip, invert) in enumerate(params):
            replace = r, g, b, a
            if rgba is not None:
                rgba = image_split(tensor_to_cv(rgba, chan=4))
                img = [tensor_to_cv(replace[i]) if replace[i] is not None else x for i, x in enumerate(rgba)]
            else:
                img = [tensor_to_cv(x) if x is not None else x for x in replace]

            _, _, w_max, h_max = image_minmax(img)
            for i, x in enumerate(img):
                if x is None:
                    x = np.full((h_max, w_max, 1), matte[i], dtype=np.uint8)
                else:
                    x = image_convert(x, 1)
                    x = image_scalefit(x, w_max, h_max, EnumScaleMode.ASPECT)

                if flip[i] != 0:
                    x = image_invert(x, flip[i])
                img[i] = x

            img = channel_merge(img)

            #if invert == True:
            #    img = image_invert(img, 1)

            images.append(cv_to_tensor_full(img, matte))
            pbar.update_absolute(idx)
        return image_stack(images)

class PixelSplitNode(CozyBaseNode):
    NAME = "PIXEL SPLIT (JOV) 💔"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("❤️", "💚", "💙", "🤍", "RGB")
    OUTPUT_TOOLTIPS = (
        "Single channel output of Red Channel.",
        "Single channel output of Green Channel",
        "Single channel output of Blue Channel",
        "Single channel output of Alpha Channel",
        "RGB pack of the input",
    )
    DESCRIPTION = """
Split an input into individual color channels (red, green, blue, alpha).
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        images = []
        pbar = ProgressBar(len(pA))
        for idx, pA in enumerate(pA):
            pA = channel_solid(chan=EnumImageType.RGBA) if pA is None else tensor_to_cv(pA, chan=4)
            out = [cv_to_tensor(x, True) for x in image_split(pA)] + [cv_to_tensor(image_convert(pA, 3))]
            images.append(out)
            pbar.update_absolute(idx)
        return image_stack(images)

class PixelSwapNode(CozyImageNode):
    NAME = "PIXEL SWAP (JOV) 🔃"
    CATEGORY = JOV_CATEGORY
    DESCRIPTION = """
Swap pixel values between two input images based on specified channel swizzle operations. Options include pixel inputs, swap operations for red, green, blue, and alpha channels, and constant values for each channel. The swap operations allow for flexible pixel manipulation by determining the source of each channel in the output image, whether it be from the first image, the second image, or a constant value.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE_SOURCE: (COZY_TYPE_IMAGE, {}),
                Lexicon.IMAGE_TARGET: (COZY_TYPE_IMAGE, {}),
                Lexicon.SWAP_R: (EnumPixelSwizzle._member_names_, {
                    "default": EnumPixelSwizzle.RED_A.name,}),
                Lexicon.SWAP_G: (EnumPixelSwizzle._member_names_, {
                    "default": EnumPixelSwizzle.GREEN_A.name,}),
                Lexicon.SWAP_B: (EnumPixelSwizzle._member_names_, {
                    "default": EnumPixelSwizzle.BLUE_A.name,}),
                Lexicon.SWAP_A: (EnumPixelSwizzle._member_names_, {
                    "default": EnumPixelSwizzle.ALPHA_A.name,}),
                Lexicon.MATTE: ("VEC4", {
                    "default": (0, 0, 0, 255), "rgb": True,})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE_SOURCE, EnumConvertType.IMAGE, None)
        pB = parse_param(kw, Lexicon.IMAGE_TARGET, EnumConvertType.IMAGE, None)
        swap_r = parse_param(kw, Lexicon.SWAP_R, EnumPixelSwizzle, EnumPixelSwizzle.RED_A.name)
        swap_g = parse_param(kw, Lexicon.SWAP_G, EnumPixelSwizzle, EnumPixelSwizzle.GREEN_A.name)
        swap_b = parse_param(kw, Lexicon.SWAP_B, EnumPixelSwizzle, EnumPixelSwizzle.BLUE_A.name)
        swap_a = parse_param(kw, Lexicon.SWAP_A, EnumPixelSwizzle, EnumPixelSwizzle.ALPHA_A.name)
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, (0, 0, 0, 255), 0, 255)
        params = list(zip_longest_fill(pA, pB, swap_r, swap_g, swap_b, swap_a, matte))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, pB, swap_r, swap_g, swap_b, swap_a, matte) in enumerate(params):
            if pA is None:
                if pB is None:
                    out = channel_solid()
                    images.append(cv_to_tensor_full(out))
                    pbar.update_absolute(idx)
                    continue

                h, w = pB.shape[:2]
                pA = channel_solid(w, h)
            else:
                h, w = pA.shape[:2]
                pA = tensor_to_cv(pA)
                pA = image_convert(pA, 4)

            pB = tensor_to_cv(pB) if pB is not None else channel_solid(w, h)
            pB = image_convert(pB, 4)
            pB = image_matte(pB, (0,0,0,0), w, h)
            pB = image_scalefit(pB, w, h, EnumScaleMode.CROP)

            out = channel_swap(pA, pB, (swap_r, swap_g, swap_b, swap_a), matte)

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
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.ADAPT: ( EnumThresholdAdapt._member_names_, {
                    "default": EnumThresholdAdapt.ADAPT_NONE.name,}),
                Lexicon.FUNCTION: ( EnumThreshold._member_names_, {
                    "default": EnumThreshold.BINARY.name}),
                Lexicon.THRESHOLD: ("FLOAT", {
                    "default": 0.5, "min": 0, "max": 1, "step": 0.005}),
                Lexicon.SIZE: ("INT", {
                    "default": 3, "min": 3, "max": 103}),
                Lexicon.INVERT: ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask input"})
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> RGBAMaskType:
        pA = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        mode = parse_param(kw, Lexicon.FUNCTION, EnumThreshold, EnumThreshold.BINARY.name)
        adapt = parse_param(kw, Lexicon.ADAPT, EnumThresholdAdapt, EnumThresholdAdapt.ADAPT_NONE.name)
        threshold = parse_param(kw, Lexicon.THRESHOLD, EnumConvertType.FLOAT, 1, 0, 1)
        block = parse_param(kw, Lexicon.SIZE, EnumConvertType.INT, 3, 3, 103)
        invert = parse_param(kw, Lexicon.INVERT, EnumConvertType.BOOLEAN, False)
        params = list(zip_longest_fill(pA, mode, adapt, threshold, block, invert))
        images = []
        pbar = ProgressBar(len(params))
        for idx, (pA, mode, adapt, th, block, invert) in enumerate(params):
            pA = tensor_to_cv(pA) if pA is not None else channel_solid()
            pA = image_threshold(pA, th, mode, adapt, block)
            if invert == True:
                pA = image_invert(pA, 1)
            images.append(cv_to_tensor_full(pA))
            pbar.update_absolute(idx)
        return image_stack(images)
