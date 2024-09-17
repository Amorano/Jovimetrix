"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Extras Support
"""

import math
import sys
from typing import Any, List, Tuple

import cv2
import numpy as np
from numba import jit
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageChops

from loguru import logger

from Jovimetrix.sup.util import grid_make

from Jovimetrix.sup.image import TAU, TYPE_IMAGE, TYPE_PIXEL, TYPE_fCOORD2D, \
    TYPE_iRGB, EnumImageBySize, EnumMirrorMode, EnumOrientation, \
    EnumThreshold, EnumThresholdAdapt, bgr2image, channel_add, cv2pil, \
    image2bgr, image_grayscale, image_matte, image_normalize, pil2cv, \
    pixel_convert, image_convert

# =============================================================================
# === EXPLICIT SHAPE FUNCTIONS ===
# =============================================================================

def shape_ellipse(width: int, height: int, sizeX:float=1., sizeY:float=1.,
                  fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    ImageDraw.Draw(image).ellipse(xy, fill=fill)
    return image

def shape_quad(width: int, height: int, sizeX:float=1., sizeY:float=1.,
               fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    ImageDraw.Draw(image).rectangle(xy, fill=fill)
    return image

def shape_polygon(width: int, height: int, size: float=1., sides: int=3,
                  fill:TYPE_PIXEL=255, back:TYPE_PIXEL=0) -> Image:
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), back)
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image

def image_by_size(image_list: List[TYPE_IMAGE],
                  enumSize: EnumImageBySize=EnumImageBySize.LARGEST) -> Tuple[TYPE_IMAGE, int, int]:

    img = None
    mega, width, height = 0, 0, 0
    if enumSize in [EnumImageBySize.SMALLEST, EnumImageBySize.WIDTH_MIN, EnumImageBySize.HEIGHT_MIN]:
        mega, width, height = sys.maxsize, sys.maxsize, sys.maxsize

    for i in image_list:
        h, w = i.shape[:2]
        match enumSize:
            case EnumImageBySize.LARGEST:
                if (new_mega := w * h) > mega:
                    mega = new_mega
                    img = i
                width = max(width, w)
                height = max(height, h)
            case EnumImageBySize.SMALLEST:
                if (new_mega := w * h) < mega:
                    mega = new_mega
                    img = i
                width = min(width, w)
                height = min(height, h)
            case EnumImageBySize.WIDTH_MIN:
                if w < width:
                    width = w
                    img = i
            case EnumImageBySize.WIDTH_MAX:
                if w > width:
                    width = w
                    img = i
            case EnumImageBySize.HEIGHT_MIN:
                if h < height:
                    height = h
                    img = i
            case EnumImageBySize.HEIGHT_MAX:
                if h > height:
                    height = h
                    img = i

    return img, width, height

def image_detect(image: TYPE_IMAGE) -> Tuple[TYPE_IMAGE, Tuple[int, ...]]:
    gray = image_grayscale(image)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the item we want to recenter
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image, (x, y, w, h)

def image_diff(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, threshold:int=0,
               color:TYPE_PIXEL=(255, 0, 0)) -> Tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, float]:
    """imageA, imageB, diff, thresh, score
    """
    h1, w1 = imageA.shape[:2]
    h2, w2 = imageB.shape[:2]
    w1 = max(w1, w2)
    h1 = max(h1, h2)
    imageA = image_matte(imageA, (0, 0, 0, 0), w1, h1)
    imageA = image_convert(imageA, 3)
    imageB = image_matte(imageB, (0, 0, 0, 0), w1, h1)
    imageB = image_convert(imageB, 3)
    grayA = image_grayscale(imageA)
    grayB = image_grayscale(imageB)
    (score, diff) = ssim(grayA, grayB, full=True, channel_axis=2)
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    high_a = imageA.copy()
    high_a = image_convert(high_a, 3)
    high_b = imageB.copy()
    high_b = image_convert(high_b, 3)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(high_a, [c], 0, color[::-1], -1)
        cv2.drawContours(high_b, [c], 0, color[::-1], -1)
        cv2.drawContours(diff_box, [c], 0, color[::-1], -1)
    imageA = cv2.addWeighted(imageA, 0.0, high_a, 1, 0)
    imageB = cv2.addWeighted(imageB, 0.0, high_b, 1, 0)
    return imageA, imageB, diff, thresh, score

def image_disparity(imageA: np.ndarray) -> np.ndarray:
    imageA = imageA.astype(np.float32) / 255.
    imageA = cv2.normalize(imageA, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    disparity_map = np.divide(1.0, imageA, where=imageA != 0)
    return np.where(imageA == 0, 1, disparity_map)

def image_gradient_map2(image, gradient_map):
    na = np.array(image)
    grey = np.mean(na, axis=2).astype(np.uint8)
    cmap = np.array(gradient_map.convert('RGB'))
    result = np.zeros((*grey.shape, 3), dtype=np.uint8)
    grey_reshaped = grey.reshape(-1)
    np.take(cmap.reshape(-1, 3), grey_reshaped, axis=0, out=result.reshape(-1, 3))
    return result

def image_gradient(width:int, height:int, color_map:dict=None) -> TYPE_IMAGE:
    if color_map is None:
        color_map = {0: (0,0,0,255)}
    else:
        color_map = {np.clip(float(k), 0, 1): [np.clip(int(c), 0, 255) for c in v] for k, v in color_map.items()}
    color_map = dict(sorted(color_map.items()))
    image = Image.new('RGBA', (width, height))
    draw = image.load()
    widthf = float(width)

    @jit
    def gaussian(x, a, b, c, d=0) -> Any:
        return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

    def pixel(x, spread:int=1) -> TYPE_iRGB:
        ws = widthf / (spread * len(color_map))
        r = sum([gaussian(x, p[0], k * widthf, ws) for k, p in color_map.items()])
        g = sum([gaussian(x, p[1], k * widthf, ws) for k, p in color_map.items()])
        b = sum([gaussian(x, p[2], k * widthf, ws) for k, p in color_map.items()])
        return min(255, int(r)), min(255, int(g)), min(255, int(b))

    for x in range(width):
        r, g, b = pixel(x)
        for y in range(height):
            draw[x, y] = r, g, b
    return pil2cv(image)

# Adapted from WAS Suite -- gradient_map
# https://github.com/WASasquatch/was-node-suite-comfyui
def image_gradient_map(image:TYPE_IMAGE, gradient_map:TYPE_IMAGE, reverse:bool=False) -> TYPE_IMAGE:
    if reverse:
        gradient_map = gradient_map[:,:,::-1]
    grey = image_grayscale(image)
    cmap = image_convert(gradient_map, 3)
    cmap = cv2.resize(cmap, (256, 256))
    cmap = cmap[0,:,:].reshape((256, 1, 3)).astype(np.uint8)
    return cv2.applyColorMap(grey, cmap)

def image_grid(data: List[TYPE_IMAGE], width: int, height: int) -> TYPE_IMAGE:
    #@TODO: makes poor assumption all images are the same dimensions.
    chunks, col, row = grid_make(data)
    frame = np.zeros((height * row, width * col, 4), dtype=np.uint8)
    i = 0
    for y, strip in enumerate(chunks):
        for x, item in enumerate(strip):
            cc = item.shape[2] if item.ndim == 3 else 1
            if cc == 3:
                item = channel_add(item)
            y1, y2 = y * height, (y+1) * height
            x1, x2 = x * width, (x+1) * width
            frame[y1:y2, x1:x2, ] = item
            i += 1

    return frame

def image_histogram(image:TYPE_IMAGE, bins=256) -> TYPE_IMAGE:
    bins = max(image.max(), bins) + 1
    flatImage = image.flatten()
    histogram = np.zeros(bins)
    for pixel in flatImage:
        histogram[pixel] += 1
    return histogram

def image_lerp(imageA: TYPE_IMAGE, imageB:TYPE_IMAGE, mask:TYPE_IMAGE=None,
               alpha:float=1.) -> TYPE_IMAGE:

    imageA = imageA.astype(np.float32)
    imageB = imageB.astype(np.float32)

    # establish mask
    alpha = np.clip(alpha, 0, 1)
    if mask is None:
        height, width = imageA.shape[:2]
        mask = np.ones((height, width, 1), dtype=np.float32)
    else:
        # normalize the mask
        mask = mask.astype(np.float32)
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * alpha

    # LERP
    imageA = cv2.multiply(1. - mask, imageA)
    imageB = cv2.multiply(mask, imageB)
    imageA = (cv2.add(imageA, imageB) / 255. - 0.5) * 2.0
    imageA = (imageA * 255).astype(np.uint8)
    return np.clip(imageA, 0, 255)

def image_levels(image: np.ndarray, black_point:int=0, white_point=255,
        mid_point=128, gamma=1.0) -> np.ndarray:
    """
    Adjusts the levels of an image including black, white, midpoints, and gamma correction.

    Args:
        image (numpy.ndarray): Input image tensor in RGB(A) format.
        black_point (int): The black point to adjust shadows. Default is 0.
        white_point (int): The white point to adjust highlights. Default is 255.
        mid_point (int): The mid point for mid-tone adjustment. Default is 128.
        gamma (float): Gamma correction value. Default is 1.0.

    Returns:
        numpy.ndarray: Adjusted image tensor.
    """

    image, alpha, cc = image2bgr(image)

    # Convert points and gamma to float32 for calculations
    black = np.array([black_point] * 3, dtype=np.float32)
    white = np.array([white_point] * 3, dtype=np.float32)
    mid = np.array([mid_point] * 3, dtype=np.float32)
    inGamma = np.array([gamma] * 3, dtype=np.float32)
    outBlack = np.array([0, 0, 0], dtype=np.float32)
    outWhite = np.array([255, 255, 255], dtype=np.float32)

    # Apply levels adjustment
    image = np.clip((image - black) / (white - black), 0, 1)
    image = (image - mid) / (1.0 - mid)
    image = (image ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    image = np.clip(image, 0, 255).astype(np.uint8)
    return bgr2image(image, alpha, cc == 1)

def image_merge(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, axis: int=0,
                flip: bool=False) -> TYPE_IMAGE:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def image_mirror(image: TYPE_IMAGE, mode:EnumMirrorMode, x:float=0.5,
                 y:float=0.5) -> TYPE_IMAGE:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]

    def mirror(img:TYPE_IMAGE, axis:int, reverse:bool=False) -> TYPE_IMAGE:
        pivot = x if axis == 1 else y
        flip = cv2.flip(img, axis)
        pivot = np.clip(pivot, 0, 1)
        if reverse:
            pivot = 1. - pivot
            flip, img = img, flip

        scalar = height if axis == 0 else width
        slice1 = int(pivot * scalar)
        slice1w = scalar - slice1
        slice2w = min(scalar - slice1w, slice1w)

        if cc >= 3:
            output = np.zeros((height, width, cc), dtype=np.uint8)
        else:
            output = np.zeros((height, width), dtype=np.uint8)

        if axis == 0:
            output[:slice1, :] = img[:slice1, :]
            output[slice1:slice1 + slice2w, :] = flip[slice1w:slice1w + slice2w, :]
        else:
            output[:, :slice1] = img[:, :slice1]
            output[:, slice1:slice1 + slice2w] = flip[:, slice1w:slice1w + slice2w]

        return output

    if mode in [EnumMirrorMode.X, EnumMirrorMode.FLIP_X, EnumMirrorMode.XY, EnumMirrorMode.FLIP_XY, EnumMirrorMode.X_FLIP_Y, EnumMirrorMode.FLIP_X_FLIP_Y]:
        reverse = mode in [EnumMirrorMode.FLIP_X, EnumMirrorMode.FLIP_XY, EnumMirrorMode.FLIP_X_FLIP_Y]
        image = mirror(image, 1, reverse)

    if mode not in [EnumMirrorMode.NONE, EnumMirrorMode.X, EnumMirrorMode.FLIP_X]:
        reverse = mode in [EnumMirrorMode.FLIP_Y, EnumMirrorMode.FLIP_X_FLIP_Y, EnumMirrorMode.X_FLIP_Y]
        image = mirror(image, 0, reverse)

    return image

def image_mirror_mandela(imageA: np.ndarray, imageB: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Merge 4 flipped copies of input images to make them wrap.
    Output is twice bigger in both dimensions."""

    top = np.hstack([imageA, -np.flip(imageA, axis=1)])
    bottom = np.hstack([np.flip(imageA, axis=0), -np.flip(imageA)])
    imageA = np.vstack([top, bottom])

    top = np.hstack([imageB, np.flip(imageB, axis=1)])
    bottom = np.hstack([-np.flip(imageB, axis=0), -np.flip(imageB)])
    imageB = np.vstack([top, bottom])
    return imageA, imageB

def image_pixelate(image: TYPE_IMAGE, amount:float=1.)-> TYPE_IMAGE:

    h, w = image.shape[:2]
    amount = max(0, min(1, amount))
    block_size_h = max(1, (h * amount))
    block_size_w = max(1, (w * amount))
    num_blocks_h = int(np.ceil(h / block_size_h))
    num_blocks_w = int(np.ceil(w / block_size_w))
    block_size_h = h // num_blocks_h
    block_size_w = w // num_blocks_w
    pixelated_image = image.copy()

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Calculate block boundaries
            y_start = i * block_size_h
            y_end = min((i + 1) * block_size_h, h)
            x_start = j * block_size_w
            x_end = min((j + 1) * block_size_w, w)

            # Average color values within the block
            block_average = np.mean(image[y_start:y_end, x_start:x_end], axis=(0, 1))

            # Fill the block with the average color
            pixelated_image[y_start:y_end, x_start:x_end] = block_average

    return pixelated_image.astype(np.uint8)

def image_posterize(image: TYPE_IMAGE, levels:int=256) -> TYPE_IMAGE:
    divisor = 256 / max(2, min(256, levels))
    return (np.floor(image / divisor) * int(divisor)).astype(np.uint8)

def image_quantize(image:TYPE_IMAGE, levels:int=256, iterations:int=10,
                   epsilon:float=0.2) -> TYPE_IMAGE:
    levels = int(max(2, min(256, levels)))
    pixels = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    _, labels, centers = cv2.kmeans(pixels, levels, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(image.shape)

def image_sharpen(image:TYPE_IMAGE, kernel_size=None, sigma:float=1.0,
                amount:float=1.0, threshold:float=0) -> TYPE_IMAGE:
    """Return a sharpened version of the image, using an unsharp mask."""

    kernel_size = (kernel_size, kernel_size) if kernel_size else (5, 5)
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def image_split(image: TYPE_IMAGE) -> Tuple[TYPE_IMAGE, ...]:
    h, w = image.shape[:2]

    # Grayscale image
    if image.ndim == 2 or image.shape[2] == 1:
        r = g = b = image
        a = np.full((h, w), 255, dtype=image.dtype)

    # BGR image
    elif image.shape[2] == 3:
        r, g, b = cv2.split(image)
        a = np.full((h, w), 255, dtype=image.dtype)
    else:
        r, g, b, a = cv2.split(image)
    return r, g, b, a

def image_stack(image_list: List[TYPE_IMAGE],
                axis:EnumOrientation=EnumOrientation.HORIZONTAL,
                stride:int=0, matte:TYPE_PIXEL=(0,0,0,255)) -> TYPE_IMAGE:

    _, width, height = image_by_size(image_list)
    images = [image_matte(image_convert(i, 4), matte, width, height) for i in image_list]
    count = len(images)

    matte = pixel_convert(matte, 4)
    match axis:
        case EnumOrientation.GRID:
            if stride < 1:
                stride = np.ceil(np.sqrt(count))
                stride = int(stride)
            stride = min(stride, count)
            stride = max(stride, 1)

            rows = []
            for i in range(0, count, stride):
                row = images[i:i + stride]
                row_stacked = np.hstack(row)
                rows.append(row_stacked)

            height, width = images[0].shape[:2]
            overhang = count % stride
            if overhang != 0:
                overhang = stride - overhang
                size = (height, overhang * width, 4)
                filler = np.full(size, matte, dtype=np.uint8)
                rows[-1] = np.hstack([rows[-1], filler])
            image = np.vstack(rows)

        case EnumOrientation.HORIZONTAL:
            image = np.hstack(images)

        case EnumOrientation.VERTICAL:
            image = np.vstack(images)
    return image

def image_stereogram(image: TYPE_IMAGE, depth: TYPE_IMAGE, divisions:int=8,
                     mix:float=0.33, gamma:float=0.33, shift:float=1.) -> TYPE_IMAGE:
    height, width = depth.shape[:2]
    out = np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.resize(image, (width, height))
    image = image_convert(image, 3)
    depth = image_convert(depth, 3)
    noise = np.random.randint(0, max(1, int(gamma * 255)), (height, width, 3), dtype=np.uint8)
    # noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    image = cv2.addWeighted(image, 1. - mix, noise, mix, 0)

    pattern_width = width // divisions
    # shift -= 1
    for y in range(height):
        for x in range(width):
            if x < pattern_width:
                out[y, x] = image[y, x]
            else:
                # out[y, x] = out[y, x - pattern_width + int(shift * invert)]
                offset = depth[y, x][0] // divisions
                pos = x - pattern_width + int(shift * offset)
                # pos = max(-pattern_width, min(pattern_width, pos))
                out[y, x] = out[y, pos]
    return out

def image_stereo_shift(image: TYPE_IMAGE, depth: TYPE_IMAGE, shift:float=10) -> TYPE_IMAGE:
    # Ensure base image has alpha
    image = image_convert(image, 4)
    depth = image_convert(depth, 1)
    deltas = np.array((depth / 255.0) * float(shift), dtype=int)
    shifted_data = np.zeros(image.shape, dtype=np.uint8)
    _, width = image.shape[:2]
    for y, row in enumerate(deltas):
        for x, dx in enumerate(row):
            x2 = x + dx
            if (x2 >= width) or (x2 < 0):
                continue
            shifted_data[y][x2] = image[y][x]

    shifted_image = cv2pil(shifted_data)
    alphas_image = Image.fromarray(
        ndimage.binary_fill_holes(
            ImageChops.invert(
                shifted_image.getchannel("A")
            )
        )
    ).convert("1")
    shifted_image.putalpha(ImageChops.invert(alphas_image))
    return pil2cv(shifted_image)

def image_threshold(image:TYPE_IMAGE, threshold:float=0.5,
                    mode:EnumThreshold=EnumThreshold.BINARY,
                    adapt:EnumThresholdAdapt=EnumThresholdAdapt.ADAPT_NONE,
                    block:int=3, const:float=0.) -> TYPE_IMAGE:

    const = max(-100, min(100, const))
    block = max(3, block if block % 2 == 1 else block + 1)
    image, alpha, cc = image2bgr(image)
    if adapt != EnumThresholdAdapt.ADAPT_NONE:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, adapt.value, cv2.THRESH_BINARY, block, const)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # gray = np.stack([gray, gray, gray], axis=-1)
        image = cv2.bitwise_and(image, gray)
    else:
        threshold = int(threshold * 255)
        _, image = cv2.threshold(image, threshold, 255, mode.value)
    return bgr2image(image, alpha, cc == 1)

# MORPHOLOGY

def morph_edge_detect(image: TYPE_IMAGE,
                    ksize: int=3,
                    low: float=0.27,
                    high:float=0.6) -> TYPE_IMAGE:

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ksize = max(3, ksize)
    image = cv2.GaussianBlur(src=image, ksize=(ksize, ksize+2), sigmaX=0.5)
    # Perform Canny edge detection
    return cv2.Canny(image, int(low * 255), int(high * 255))

def morph_emboss(image: TYPE_IMAGE, amount: float=1., kernel: int=2) -> TYPE_IMAGE:
    kernel = max(2, kernel)
    kernel = np.array([
        [-kernel,   -kernel+1,    0],
        [-kernel+1,   kernel-1,     1],
        [kernel-2,    kernel-1,     2]
    ]) * amount
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# KERNELS

def MEDIAN3x3(image: TYPE_IMAGE) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    out = np.zeros([height, width])
    for i in range(1, height-1):
        for j in range(1, width-1):
            temp = [
                image[i-1, j-1],
                image[i-1, j],
                image[i-1, j + 1],
                image[i, j-1],
                image[i, j],
                image[i, j + 1],
                image[i + 1, j-1],
                image[i + 1, j],
                image[i + 1, j + 1]
            ]

            temp = sorted(temp)
            out[i, j]= temp[4]
    return out

def kernel(stride: int) -> TYPE_IMAGE:
    """
    Generate a kernel matrix with a specific stride.

    The kernel matrix has a size of (stride, stride) and is filled with values
    such that if i < j, the element is set to -1; if i > j, the element is set to 1.

    Parameters:
    - stride (int): The size of the square kernel matrix.

    Returns:
    - TYPE_IMAGE: The generated kernel matrix.

    Example:
    >>> KERNEL(3)
    array([[ 0, 1, 1],
        [-1, 0, 1],
        [-1, -1, 0]], dtype=int8)
    """
    # Create an initial matrix of zeros
    kernel = np.zeros((stride, stride), dtype=np.int8)

    # Create a mask for elements where i < j and set them to -1
    mask_lower = np.tril(np.ones((stride, stride), dtype=bool), k=-1)
    kernel[mask_lower] = -1

    # Create a mask for elements where i > j and set them to 1
    mask_upper = np.triu(np.ones((stride, stride), dtype=bool), k=1)
    kernel[mask_upper] = 1

    return kernel

# =============================================================================

def coord_cart2polar(x: float, y: float) -> TYPE_fCOORD2D:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def coord_polar2cart(r: float, theta: float) -> TYPE_fCOORD2D:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def coord_default(width:int, height:int, origin:TYPE_fCOORD2D=None) -> TYPE_fCOORD2D:
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    if origin is None:
        origin_x, origin_y = width // 2, height // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x -= origin_x
    y -= origin_y
    return x, y

def coord_fisheye(width: int, height: int, distortion: float) -> Tuple[TYPE_IMAGE, TYPE_IMAGE]:
    map_x, map_y = np.meshgrid(np.linspace(0., 1., width), np.linspace(0., 1., height))
    # normalized
    xnd, ynd = (2 * map_x - 1), (2 * map_y - 1)
    rd = np.sqrt(xnd**2 + ynd**2)
    # fish-eye distortion
    condition = (dist := 1 - distortion * (rd**2)) == 0
    xdu, ydu = np.where(condition, xnd, xnd / dist), np.where(condition, ynd, ynd / dist)
    xu, yu = ((xdu + 1) * width) / 2, ((ydu + 1) * height) / 2
    return xu.astype(np.float32), yu.astype(np.float32)

def coord_perspective(width: int, height: int, pts: List[TYPE_fCOORD2D]) -> TYPE_IMAGE:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0], pts[:, 1]])
    return cv2.getPerspectiveTransform(object_pts, pts)

def coord_sphere(width: int, height: int, radius: float) -> Tuple[TYPE_IMAGE, TYPE_IMAGE]:
    theta, phi = np.meshgrid(np.linspace(0, TAU, width), np.linspace(0, np.pi, height))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    # z = radius * np.cos(phi)
    x_image = (x + 1) * (width - 1) / 2
    y_image = (y + 1) * (height - 1) / 2
    return x_image.astype(np.float32), y_image.astype(np.float32)

def remap_fisheye(image: TYPE_IMAGE, distort: float) -> TYPE_IMAGE:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    map_x, map_y = coord_fisheye(width, height, distort)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #if cc == 1:
    #    image = image[..., 0]
    return image

def remap_perspective(image: TYPE_IMAGE, pts: list) -> TYPE_IMAGE:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pts = coord_perspective(width, height, pts)
    image = cv2.warpPerspective(image, pts, (width, height))
    #if cc == 1:
    #    image = image[..., 0]
    return image

def remap_polar(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Re-projects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    h, w = image.shape[:2]
    radius = max(w, h)
    return cv2.linearPolar(image, (h // 2, w // 2), radius // 2, cv2.WARP_INVERSE_MAP)

def remap_sphere(image: TYPE_IMAGE, radius: float) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    map_x, map_y = coord_sphere(width, height, radius)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def depth_from_gradient(grad_x, grad_y):
    """Optimized Frankot-Chellappa depth-from-gradient algorithm."""
    rows, cols = grad_x.shape
    rows_scale = np.fft.fftfreq(rows)
    cols_scale = np.fft.fftfreq(cols)
    u_grid, v_grid = np.meshgrid(cols_scale, rows_scale)
    grad_x_F = np.fft.fft2(grad_x)
    grad_y_F = np.fft.fft2(grad_y)
    denominator = u_grid**2 + v_grid**2
    denominator[0, 0] = 1.0
    Z_F = (-1j * u_grid * grad_x_F - 1j * v_grid * grad_y_F) / denominator
    Z_F[0, 0] = 0.0
    Z = np.fft.ifft2(Z_F).real
    Z -= np.min(Z)
    Z /= np.max(Z)
    return Z

def height_from_normal(image: TYPE_IMAGE, tile:bool=True) -> TYPE_IMAGE:
    """Computes a height map from the given normal map."""
    image = np.transpose(image, (2, 0, 1))
    flip_img = np.flip(image, axis=1)
    grad_x, grad_y = (flip_img[0] - 0.5) * 2, (flip_img[1] - 0.5) * 2
    grad_x = np.flip(grad_x, axis=0)
    grad_y = np.flip(grad_y, axis=0)

    if not tile:
        grad_x, grad_y = image_mirror_mandela(grad_x, grad_y)
    pred_img = depth_from_gradient(-grad_x, grad_y)

    # re-crop
    if not tile:
        height, width = image.shape[1], image.shape[2]
        pred_img = pred_img[:height, :width]

    image = np.stack([pred_img, pred_img, pred_img])
    image = np.transpose(image, (1, 2, 0))
    return image

def curvature_from_normal(image: TYPE_IMAGE, blur_radius:int=2)-> TYPE_IMAGE:
    """Computes a curvature map from the given normal map."""
    image = np.transpose(image, (2, 0, 1))
    blur_factor = 1 / 2 ** min(8, max(2, blur_radius))
    diff_kernel = np.array([-1, 0, 1])

    def conv_1d(array, kernel) -> np.ndarray[Any, np.dtype[Any]]:
        """Performs row-wise 1D convolutions with repeat padding."""
        k_l = len(kernel)
        extended = np.pad(array, k_l // 2, mode="wrap")
        return np.array([np.convolve(row, kernel, mode="valid") for row in extended[k_l//2:-k_l//2+1]])

    h_conv = conv_1d(image[0], diff_kernel)
    v_conv = conv_1d(-image[1].T, diff_kernel).T
    edges_conv = h_conv + v_conv

    # Calculate blur radius in pixels
    blur_radius_px = int(np.mean(image.shape[1:3]) * blur_factor)
    if blur_radius_px < 2:
        # If blur radius is too small, just normalize the edge convolution
        image = (edges_conv - np.min(edges_conv)) / (np.ptp(edges_conv) + 1e-10)
    else:
        blur_radius_px += blur_radius_px % 2 == 0

        # Compute Gaussian kernel
        sigma = max(1, blur_radius_px // 8)
        x = np.linspace(-(blur_radius_px - 1) / 2, (blur_radius_px - 1) / 2, blur_radius_px)
        g_kernel = np.exp(-0.5 * np.square(x) / np.square(sigma))
        g_kernel /= np.sum(g_kernel)

        # Apply Gaussian blur
        h_blur = conv_1d(edges_conv, g_kernel)
        v_blur = conv_1d(h_blur.T, g_kernel).T
        image = (v_blur - np.min(v_blur)) / (np.ptp(v_blur) + 1e-10)

    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.astype(np.uint8)

def roughness_from_normal(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Roughness from a normal map."""
    up_vector = np.array([0, 0, 1])
    image = 1 - np.dot(image, up_vector)
    image = (image - image.min()) / (image.max() - image.min())
    image = (255 * image).astype(np.uint8)
    return image_grayscale(image)

def roughness_from_albedo(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Roughness from an albedo map."""
    kernel_size = 3
    image = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    image = (image - image.min()) / (image.max() - image.min())
    image = (255 * image).astype(np.uint8)
    return image_grayscale(image)

def roughness_from_albedo_normal(albedo: TYPE_IMAGE, normal: TYPE_IMAGE,
                                 blur:int=2, blend:float=0.5, iterations:int=3) -> TYPE_IMAGE:
    normal = roughness_from_normal(normal)
    normal = image_normalize(normal)
    albedo = roughness_from_albedo(albedo)
    albedo = image_normalize(albedo)
    rough = image_lerp(normal, albedo, alpha=blend)
    rough = image_normalize(rough)
    image = image_lerp(normal, rough, alpha=blend)
    iterations = min(16, max(2, iterations))
    blur += (blur % 2 == 0)
    step = 1 / 2 ** iterations
    for i in range(iterations):
        image = cv2.add(normal * step, image * step)
        image = cv2.GaussianBlur(image, (blur + i * 2, blur + i * 2), 3 * i)

    inverted = 255 - image_normalize(image)
    inverted = cv2.subtract(inverted, albedo) * 0.5
    inverted = cv2.GaussianBlur(inverted, (blur, blur), blur)
    inverted = image_normalize(inverted)

    image = cv2.add(image * 0.5, inverted * 0.5)
    for i in range(iterations):
        image = cv2.GaussianBlur(image, (blur, blur), blur)

    image = cv2.add(image * 0.5, inverted * 0.5)
    for i in range(iterations):
        image = cv2.GaussianBlur(image, (blur, blur), blur)

    image = image_normalize(image)
    return image
