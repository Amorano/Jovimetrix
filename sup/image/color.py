"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Image Color Support
"""

from typing import Tuple

import cv2
import cupy as cp
import numpy as np
from numba import cuda
from daltonlens import simulate
from skimage import exposure
from blendmodes.blend import BlendType

from Jovimetrix.sup.image import TYPE_IMAGE, TYPE_PIXEL, EnumCBDeficiency, \
    EnumCBSimulator, EnumColorTheory, bgr2hsv, hsv2bgr, image_blend, \
    image_convert, image_mask, image_mask_add, pixel_hsv_adjust

# =============================================================================
# === COLOR FUNCTIONS ===
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
    pixels = cp.asarray(image.reshape(-1, 3)).astype(cp.float32)
    # logger.debug("Pixel range:", cp.min(pixels), cp.max(pixels))

    # Initialize centroids using random pixels
    random_indices = cp.random.choice(pixels.shape[0], size=num_colors, replace=False)
    centroids = pixels[random_indices]
    # logger.debug("Initial centroids range:", cp.min(centroids), cp.max(centroids))

    # Prepare for K-means
    assignments = cp.zeros(pixels.shape[0], dtype=cp.int32)
    threads_per_block = 256
    blocks = (pixels.shape[0] + threads_per_block - 1) // threads_per_block

    # K-means iterations
    for iteration in range(20):  # Adjust the number of iterations as needed
        kmeans_kernel[blocks, threads_per_block](pixels, centroids, assignments)
        new_centroids = cp.zeros((num_colors, 3), dtype=cp.float32)
        for i in range(num_colors):
            mask = (assignments == i)
            if cp.any(mask):
                new_centroids[i] = cp.mean(pixels[mask], axis=0)

        centroids = new_centroids

        if iteration % 5 == 0:
            # logger.debug(f"Iteration {iteration}, Centroids range: {cp.min(centroids)} {cp.max(centroids)}")
            pass

    # Create LUT
    lut = cp.zeros((256, 1, 3), dtype=cp.uint8)
    lut[:num_colors] = cp.clip(centroids, 0, 255).reshape(-1, 1, 3).astype(cp.uint8)
    # logger.debug(f"Final LUT range: { cp.min(lut)} {cp.max(lut)}")
    return cp.asnumpy(lut)

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
