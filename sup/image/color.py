"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Image Color Support
"""

from enum import Enum
from typing import Tuple

import cv2
import numpy as np
from numba import cuda
from daltonlens import simulate
from skimage import exposure
from blendmodes.blend import BlendType

from Jovimetrix.sup.image import TYPE_IMAGE, TYPE_PIXEL, bgr2hsv, hsv2bgr, \
    image_blend, image_convert, image_grayscale, image_mask, image_mask_add

# =============================================================================
# === ENUMERATION ===
# =============================================================================

class EnumColorMap(Enum):
    AUTUMN = cv2.COLORMAP_AUTUMN
    BONE = cv2.COLORMAP_BONE
    JET = cv2.COLORMAP_JET
    WINTER = cv2.COLORMAP_WINTER
    RAINBOW = cv2.COLORMAP_RAINBOW
    OCEAN = cv2.COLORMAP_OCEAN
    SUMMER = cv2.COLORMAP_SUMMER
    SPRING = cv2.COLORMAP_SPRING
    COOL = cv2.COLORMAP_COOL
    HSV = cv2.COLORMAP_HSV
    PINK = cv2.COLORMAP_PINK
    HOT = cv2.COLORMAP_HOT
    PARULA = cv2.COLORMAP_PARULA
    MAGMA = cv2.COLORMAP_MAGMA
    INFERNO = cv2.COLORMAP_INFERNO
    PLASMA = cv2.COLORMAP_PLASMA
    VIRIDIS = cv2.COLORMAP_VIRIDIS
    CIVIDIS = cv2.COLORMAP_CIVIDIS
    TWILIGHT = cv2.COLORMAP_TWILIGHT
    TWILIGHT_SHIFTED = cv2.COLORMAP_TWILIGHT_SHIFTED
    TURBO = cv2.COLORMAP_TURBO
    DEEPGREEN = cv2.COLORMAP_DEEPGREEN

class EnumColorTheory(Enum):
    COMPLIMENTARY = 0
    MONOCHROMATIC = 1
    SPLIT_COMPLIMENTARY = 2
    ANALOGOUS = 3
    TRIADIC = 4
    # TETRADIC = 5
    SQUARE = 6
    COMPOUND = 8
    # DOUBLE_COMPLIMENTARY = 9
    CUSTOM_TETRAD = 9

class EnumCBDeficiency(Enum):
    PROTAN = simulate.Deficiency.PROTAN
    DEUTAN = simulate.Deficiency.DEUTAN
    TRITAN = simulate.Deficiency.TRITAN

class EnumCBSimulator(Enum):
    AUTOSELECT = 0
    BRETTEL1997 = 1
    COBLISV1 = 2
    COBLISV2 = 3
    MACHADO2009 = 4
    VIENOT1999 = 5
    VISCHECK = 6

# ==============================================================================
# === COLOR SPACE CONVERSION ===
# ==============================================================================

def gamma2linear(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Gamma correction for old PCs/CRT monitors"""
    return np.power(image, 2.2)

def linear2gamma(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Inverse gamma correction for old PCs/CRT monitors"""
    return np.power(np.clip(image, 0., 1.), 1.0 / 2.2)

def sRGB2Linear(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Convert sRGB to linearRGB, removing the gamma correction.
    Works for grayscale, RGB, or RGBA images.
    """
    image = image.astype(float) / 255.0

    # If the image has an alpha channel, separate it
    if image.shape[-1] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3]
    else:
        rgb = image
        alpha = None

    gamma = ((rgb + 0.055) / 1.055) ** 2.4
    scale = rgb / 12.92
    rgb = np.where(rgb > 0.04045, gamma, scale)

    # Recombine the alpha channel if it exists
    if alpha is not None:
        image = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1)
    else:
        image = rgb
    return (image * 255).astype(np.uint8)

def linear2sRGB(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """Convert linearRGB to sRGB, applying the gamma correction.
    Works for grayscale, RGB, or RGBA images.
    """
    image = image.astype(float) / 255.0

    # If the image has an alpha channel, separate it
    if image.shape[-1] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3]
    else:
        rgb = image
        alpha = None

    higher = 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055
    lower = rgb * 12.92
    rgb = np.where(rgb > 0.0031308, higher, lower)

    # Recombine the alpha channel if it exists
    if alpha is not None:
        image = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1)
    else:
        image = rgb
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)

# ==============================================================================
# === PIXEL ===
# ==============================================================================

def pixel_hsv_adjust(color:TYPE_PIXEL, hue:int=0, saturation:int=0, value:int=0,
                     mod_color:bool=True, mod_sat:bool=False,
                     mod_value:bool=False) -> TYPE_PIXEL:
    """Adjust an HSV type pixel.
    OpenCV uses... H: 0-179, S: 0-255, V: 0-255"""
    hsv = [0, 0, 0]
    hsv[0] = (color[0] + hue) % 180 if mod_color else np.clip(color[0] + hue, 0, 180)
    hsv[1] = (color[1] + saturation) % 255 if mod_sat else np.clip(color[1] + saturation, 0, 255)
    hsv[2] = (color[2] + value) % 255 if mod_value else np.clip(color[2] + value, 0, 255)
    return hsv

# =============================================================================
# === COLOR MATCH ===
# =============================================================================

@cuda.jit
def kmeans_kernel(pixels, centroids, assignments) -> None:
    idx = cuda.grid(1)
    if idx < pixels.shape[0]:
        min_dist = 1e10
        min_centroid = 0
        for i in range(centroids.shape[0]):
            dist = 0
            for j in range(3):
                diff = pixels[idx, j] - centroids[i, j]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                min_centroid = i
        assignments[idx] = min_centroid

def color_image2lut(image: np.ndarray, num_colors: int = 256) -> np.ndarray:
    """Create X sized LUT from an RGB image using GPU acceleration."""
    # Ensure image is in RGB format
    if image.shape[2] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Reshape and transfer to GPU
    pixels = np.asarray(image.reshape(-1, 3)).astype(np.float32)
    # logger.debug("Pixel range:", np.min(pixels), np.max(pixels))

    # Initialize centroids using random pixels
    random_indices = np.random.choice(pixels.shape[0], size=num_colors, replace=False)
    centroids = pixels[random_indices]
    # logger.debug("Initial centroids range:", np.min(centroids), np.max(centroids))

    # Prepare for K-means
    assignments = np.zeros(pixels.shape[0], dtype=np.int32)
    threads_per_block = 256
    blocks = (pixels.shape[0] + threads_per_block - 1) // threads_per_block

    # K-means iterations
    for iteration in range(20):  # Adjust the number of iterations as needed
        kmeans_kernel[blocks, threads_per_block](pixels, centroids, assignments)
        new_centroids = np.zeros((num_colors, 3), dtype=np.float32)
        for i in range(num_colors):
            mask = (assignments == i)
            if np.any(mask):
                new_centroids[i] = np.mean(pixels[mask], axis=0)

        centroids = new_centroids

        if iteration % 5 == 0:
            # logger.debug(f"Iteration {iteration}, Centroids range: {np.min(centroids)} {np.max(centroids)}")
            pass

    # Create LUT
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[:num_colors] = np.clip(centroids, 0, 255).reshape(-1, 1, 3).astype(np.uint8)
    # logger.debug(f"Final LUT range: { np.min(lut)} {np.max(lut)}")
    return np.asnumpy(lut)

def color_blind(image: TYPE_IMAGE, deficiency:EnumCBDeficiency,
                    simulator:EnumCBSimulator=EnumCBSimulator.AUTOSELECT,
                    severity:float=1.0) -> TYPE_IMAGE:

    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 4:
        mask = image_mask(image)
    image = image_convert(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    match simulator:
        case EnumCBSimulator.AUTOSELECT:
            simulator = simulate.Simulator_AutoSelect()
        case EnumCBSimulator.BRETTEL1997:
            simulator = simulate.Simulator_Brettel1997()
        case EnumCBSimulator.COBLISV1:
            simulator = simulate.Simulator_CoblisV1()
        case EnumCBSimulator.COBLISV2:
            simulator = simulate.Simulator_CoblisV2()
        case EnumCBSimulator.MACHADO2009:
            simulator = simulate.Simulator_Machado2009()
        case EnumCBSimulator.VIENOT1999:
            simulator = simulate.Simulator_Vienot1999()
        case EnumCBSimulator.VISCHECK:
            simulator = simulate.Simulator_Vischeck()
    image = simulator.simulate_cvd(image, deficiency.value, severity=severity)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if cc == 4:
        image = image_mask_add(image, mask)
    return image

def color_match_histogram(image: TYPE_IMAGE, usermap: TYPE_IMAGE) -> TYPE_IMAGE:
    """Colorize one input based on the histogram matches."""
    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 4:
        alpha = image_mask(image)
    image = image_convert(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    beta = cv2.cvtColor(usermap, cv2.COLOR_BGR2LAB)
    image = exposure.match_histograms(image, beta, channel_axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    image = image_blend(usermap, image, blendOp=BlendType.LUMINOSITY)
    image = image_convert(image, cc)
    if cc == 4:
        image[..., 3] = alpha[..., 0]
    return image

def color_match_reinhard(image: TYPE_IMAGE, target: TYPE_IMAGE) -> TYPE_IMAGE:
    """
    Apply Reinhard color matching to an image based on a target image.
    Works only for BGR images and returns an BGR image.

    based on https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.

    Args:
        image (TYPE_IMAGE): The input image (BGR or BGRA or Grayscale).
        target (TYPE_IMAGE): The target image (BGR or BGRA or Grayscale).

    Returns:
        TYPE_IMAGE: The color-matched image in BGR format.
    """
    target = image_convert(target, 3)
    lab_tar = cv2.cvtColor(target, cv2.COLOR_BGR2Lab)
    image = image_convert(image, 3)
    lab_ori = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    mean_tar, std_tar = cv2.meanStdDev(lab_tar)
    mean_ori, std_ori = cv2.meanStdDev(lab_ori)
    ratio = (std_tar / std_ori).reshape(-1)
    offset = (mean_tar - mean_ori * std_tar / std_ori).reshape(-1)
    lab_tar = cv2.convertScaleAbs(lab_ori * ratio + offset)
    return cv2.cvtColor(lab_tar, cv2.COLOR_Lab2BGR)

def color_match_lut(image: TYPE_IMAGE, colormap:int=cv2.COLORMAP_JET,
                    usermap:TYPE_IMAGE=None, num_colors:int=255) -> TYPE_IMAGE:
    """Colorize one input based on built in cv2 color maps or a user defined image."""
    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 4:
        alpha = image_mask(image)

    image = image_convert(image, 3)
    if usermap is not None:
        usermap = image_convert(usermap, 3)
        colormap = color_image2lut(usermap, num_colors)

    image = cv2.applyColorMap(image, colormap)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, 0.5, image, 0.5, 0)
    image = image_convert(image, cc)

    if cc == 4:
        image[..., 3] = alpha[..., 0]
    return image

def color_mean(image: TYPE_IMAGE) -> TYPE_IMAGE:
    color = [0, 0, 0]
    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 1:
        raw = int(np.mean(image))
        color = [raw] * 3
    else:
        # each channel....
        color = [
            int(np.mean(image[..., 0])),
            int(np.mean(image[:,:,1])),
            int(np.mean(image[:,:,2])) ]
    return color

# ==============================================================================
# === COLOR ANALYSIS ===
# ==============================================================================

def color_theory_complementary(color: TYPE_PIXEL) -> TYPE_PIXEL:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    return hsv2bgr(color_a)

def color_theory_monochromatic(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    sat = 255 / 5
    val = 255 / 5
    color_a = pixel_hsv_adjust(color, 0, -1 * sat, -1 * val, mod_sat=True, mod_value=True)
    color_b = pixel_hsv_adjust(color, 0, -2 * sat, -2 * val, mod_sat=True, mod_value=True)
    color_c = pixel_hsv_adjust(color, 0, -3 * sat, -3 * val, mod_sat=True, mod_value=True)
    color_d = pixel_hsv_adjust(color, 0, -4 * sat, -4 * val, mod_sat=True, mod_value=True)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c), hsv2bgr(color_d)

def color_theory_split_complementary(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 75, 0, 0)
    color_b = pixel_hsv_adjust(color, 105, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b)

def color_theory_analogous(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 30, 0, 0)
    color_b = pixel_hsv_adjust(color, 15, 0, 0)
    color_c = pixel_hsv_adjust(color, 165, 0, 0)
    color_d = pixel_hsv_adjust(color, 150, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c), hsv2bgr(color_d)

def color_theory_triadic(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 60, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b)

def color_theory_compound(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    color_c = pixel_hsv_adjust(color, 150, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c)

def color_theory_square(color: TYPE_PIXEL) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 45, 0, 0)
    color_b = pixel_hsv_adjust(color, 90, 0, 0)
    color_c = pixel_hsv_adjust(color, 135, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c)

def color_theory_tetrad_custom(color: TYPE_PIXEL, delta:int=0) -> Tuple[TYPE_PIXEL, ...]:
    color = bgr2hsv(color)

    # modulus on neg and pos
    while delta < 0:
        delta += 90

    if delta > 90:
        delta = delta % 90

    color_a = pixel_hsv_adjust(color, -delta, 0, 0)
    color_b = pixel_hsv_adjust(color, delta, 0, 0)
    # just gimme a compliment
    color_c = pixel_hsv_adjust(color, 90 - delta, 0, 0)
    color_d = pixel_hsv_adjust(color, 90 + delta, 0, 0)
    return hsv2bgr(color_a), hsv2bgr(color_b), hsv2bgr(color_c), hsv2bgr(color_d)

def color_theory(image: TYPE_IMAGE, custom:int=0, scheme: EnumColorTheory=EnumColorTheory.COMPLIMENTARY) -> Tuple[TYPE_IMAGE, ...]:

    b = [0,0,0]
    c = [0,0,0]
    d = [0,0,0]
    color = color_mean(image)
    match scheme:
        case EnumColorTheory.COMPLIMENTARY:
            a = color_theory_complementary(color)
        case EnumColorTheory.MONOCHROMATIC:
            a, b, c, d = color_theory_monochromatic(color)
        case EnumColorTheory.SPLIT_COMPLIMENTARY:
            a, b = color_theory_split_complementary(color)
        case EnumColorTheory.ANALOGOUS:
            a, b, c, d = color_theory_analogous(color)
        case EnumColorTheory.TRIADIC:
            a, b = color_theory_triadic(color)
        case EnumColorTheory.SQUARE:
            a, b, c = color_theory_square(color)
        case EnumColorTheory.COMPOUND:
            a, b, c = color_theory_compound(color)
        case EnumColorTheory.CUSTOM_TETRAD:
            a, b, c, d = color_theory_tetrad_custom(color, custom)

    h, w = image.shape[:2]
    return (
        np.full((h, w, 3), color, dtype=np.uint8),
        np.full((h, w, 3), a, dtype=np.uint8),
        np.full((h, w, 3), b, dtype=np.uint8),
        np.full((h, w, 3), c, dtype=np.uint8),
        np.full((h, w, 3), d, dtype=np.uint8),
    )

#
#
#

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
