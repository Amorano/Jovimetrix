""" Jovimetrix - Image Color Support """

from enum import Enum
from typing import List

import cv2
import numpy as np
from numba import cuda
from scipy.spatial import KDTree
from skimage import exposure
from sklearn.cluster import KMeans
from daltonlens import simulate
from blendmodes.blend import BlendType

from cozy_comfyui.image import \
    PixelType, \
    EnumImageType, ImageType

from cozy_comfyui.image.convert import \
    ImageType, \
    image_mask, image_mask_add, image_convert

from cozy_comfyui.image.convert import \
    image_grayscale

from .compose import \
    image_blend

# ==============================================================================
# === TYPE ===
# ==============================================================================

TYPE_LUT = tuple[int, int, int, int]

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumIntFloat(Enum):
    FLOAT = 0
    INT = 1

class EnumGrayscaleCrunch(Enum):
    LOW = 0
    HIGH = 1
    MEAN = 2

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

def gamma2linear(image: ImageType) -> ImageType:
    """Gamma correction for old PCs/CRT monitors"""
    return np.power(image, 2.2)

def linear2gamma(image: ImageType) -> ImageType:
    """Inverse gamma correction for old PCs/CRT monitors"""
    return np.power(np.clip(image, 0., 1.), 1.0 / 2.2)

def sRGB2Linear(image: ImageType) -> ImageType:
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

def linear2sRGB(image: ImageType) -> ImageType:
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

def pixel_eval(color: PixelType,
            target: EnumImageType=EnumImageType.RGBA,
            precision:EnumIntFloat=EnumIntFloat.INT,
            crunch:EnumGrayscaleCrunch=EnumGrayscaleCrunch.MEAN) -> tuple[PixelType] | PixelType:
    """Evaluates R(GB)(A) pixels in range (0-255) into target target pixel type."""

    def parse_single_color(c: PixelType) -> PixelType:
        if not isinstance(c, int):
            c = np.clip(c, 0, 1)
            if precision == EnumIntFloat.INT:
                c = int(c * 255)
        else:
            c = np.clip(c, 0, 255)
            if precision == EnumIntFloat.FLOAT:
                c /= 255
        return c

    # make sure we are an RGBA value already
    if isinstance(color, (float, int)):
        color = tuple([parse_single_color(color)])
    elif isinstance(color, (set, tuple, list)):
        color = tuple([parse_single_color(c) for c in color])

    if target == EnumImageType.GRAYSCALE:
        alpha = 1
        if len(color) > 3:
            alpha = color[3]
            if precision == EnumIntFloat.INT:
                alpha /= 255
            color = color[:3]
        match crunch:
            case EnumGrayscaleCrunch.LOW:
                val = min(color) * alpha
            case EnumGrayscaleCrunch.HIGH:
                val = max(color) * alpha
            case EnumGrayscaleCrunch.MEAN:
                val = np.mean(color) * alpha
        if precision == EnumIntFloat.INT:
            val = int(val)
        return val

    if len(color) < 3:
        color += (0,) * (3 - len(color))

    if target in [EnumImageType.RGB, EnumImageType.BGR]:
        color = color[:3]
        if target == EnumImageType.BGR:
            color = color[::-1]
        return color

    if len(color) < 4:
        color += (255,)

    if target == EnumImageType.BGRA:
        color = tuple(color[2::-1]) + tuple([color[-1]])
    return color

def pixel_hsv_adjust(color:PixelType, hue:int=0, saturation:int=0, value:int=0,
                     mod_color:bool=True, mod_sat:bool=False,
                     mod_value:bool=False) -> PixelType:
    """Adjust an HSV type pixel.
    OpenCV uses... H: 0-179, S: 0-255, V: 0-255"""
    hsv = [0, 0, 0]
    hsv[0] = (color[0] + hue) % 180 if mod_color else np.clip(color[0] + hue, 0, 180)
    hsv[1] = (color[1] + saturation) % 255 if mod_sat else np.clip(color[1] + saturation, 0, 255)
    hsv[2] = (color[2] + value) % 255 if mod_value else np.clip(color[2] + value, 0, 255)
    return hsv

# ==============================================================================
# === COLOR MATCH ===
# ==============================================================================

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
    return np.asarray(lut)

def color_blind(image: ImageType, deficiency:EnumCBDeficiency,
                    simulator:EnumCBSimulator=EnumCBSimulator.AUTOSELECT,
                    severity:float=1.0) -> ImageType:

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

def color_lut_full(dominant_colors: List[tuple[int, int, int]], nodes:int=33) -> ImageType:
    """
    Create a 3D LUT by mapping each RGB value to the closest dominant color.
    This version is optimized for speed using vectorization.

    Args:
        dominant_colors (List[tuple[int, int, int]]): List of top colors as (R, G, B) tuples.

    Returns:
        np.ndarray: 3D LUT with shape (n, n, n, 3).
    """

    kdtree = KDTree(dominant_colors)
    r, g, b = np.mgrid[0:nodes, 0:nodes, 0:nodes]
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    _, indices = kdtree.query(rgb)
    lut = np.array(dominant_colors)[indices]
    lut = lut.reshape(nodes, nodes, nodes, 3).astype(np.uint8)
    return lut

def color_lut_match(image: ImageType, colormap:int=cv2.COLORMAP_JET,
                    usermap:ImageType=None, num_colors:int=255) -> ImageType:
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

def color_lut_palette(colors: List[tuple[int, int, int]], size: int=32) -> ImageType:
    """
    Create a color palette LUT as a 2D image from the top colors.

    Args:
        colors (List[tuple[int, int, int]]): List of top colors as (R, G, B) tuples.
        size (int): Size of each color square in the palette.

    Returns:
        np.ndarray: 2D image representing the LUT.
    """
    num_colors = len(colors)
    width = size * num_colors
    lut_image = np.zeros((size, width, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        x_start = i * size
        x_end = x_start + size
        lut_image[:, x_start:x_end] = color

    return lut_image

def color_lut_tonal(colors: List[tuple[int, int, int]], width: int=256, height: int=32) -> ImageType:
    """
    Create a 2D tonal palette LUT as a grid image from the top colors.

    Args:
        colors (List[tuple[int, int, int]]): List of top colors as (R, G, B) tuples.
        width (int): Width of each gradient row.
        height (int): Height of each color row.

    Returns:
        ImageType: 2D image representing the tonal palette LUT.
    """
    num_colors = len(colors)
    lut_image = np.zeros((height * num_colors, width, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        row_start = i * height
        row_end = row_start + height
        gradient = np.zeros((height, width, 3), dtype=np.uint8)

        for x in range(width):
            factor = x / width
            gradient[:, x] = np.array(color) * (1 - factor) + np.array([0, 0, 0]) * factor

        lut_image[row_start:row_end] = gradient

    return lut_image

def color_lut_visualize(lut: TYPE_LUT, size: int=512) -> ImageType:
    """
    Visualize a 3D LUT as a 2D image.

    Args:
        lut (np.ndarray): 3D LUT with shape (n, n, n, 3).
        size (int): Size of the output image (square). Default is 2048.

    Returns:
        PIL.Image.Image: 2D visualization of the 3D LUT.
    """
    if len(lut.shape) != 4 or lut.shape[3] != 3 or lut.shape[0] != lut.shape[1] or lut.shape[1] != lut.shape[2]:
        raise ValueError("LUT must have shape (n, n, n, 3) where n is the number of nodes per dimension")

    # 8 for a 256^3 LUT
    n = lut.shape[0]
    vis_n = int(np.ceil(np.cbrt(n)))

    # Calculate the size of each small square, ensuring it's at least 1 pixel
    square_size = max(1, size // (vis_n * vis_n))

    # Recalculate the actual image size based on the square size
    actual_size = square_size * vis_n * vis_n
    img = np.zeros((actual_size, actual_size, 3), dtype=np.uint8)

    for b in range(n):
        # Calculate position of the current slice
        slice_y = (b // vis_n) * square_size * vis_n
        slice_x = (b % vis_n) * square_size * vis_n

        # Extract the slice from the LUT
        slice_data = lut[:, :, b]
        slice_resized = cv2.resize(slice_data, (square_size * vis_n, square_size * vis_n), interpolation=cv2.INTER_NEAREST)

        # Ensure we don't go out of bounds
        end_y = min(slice_y + square_size * vis_n, actual_size)
        end_x = min(slice_x + square_size * vis_n, actual_size)
        img[slice_y:end_y, slice_x:end_x] = slice_resized[:end_y-slice_y, :end_x-slice_x]

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def color_lut_xport(lut: TYPE_LUT, f_out: str) -> None:
    """
    Save a 3D LUT as a .cube file.

    Args:
        lut (np.ndarray): 3D LUT with shape (256, 256, 256, 3).
        filename (str): Output filename (should end with .cube).
        title (str, optional): Title for the LUT. Defaults to "3D LUT".

    Returns:
        None
    """
    if lut.shape != (256, 256, 256, 3):
        raise ValueError("LUT must have shape (256, 256, 256, 3)")

    if not filename.lower().endswith('.cube'):
        filename += '.cube'

    with open(f_out, 'w') as f:
        f.write(f"TITLE 3D LUT\n")
        f.write("LUT_3D_SIZE 256\n")
        f.write("DOMAIN_MIN 0 0 0\n")
        f.write("DOMAIN_MAX 1 1 1\n\n")
        for b in range(256):
            for g in range(256):
                for r in range(256):
                    color = lut[r, g, b]
                    f.write(f"{color[0]/255:.6f} {color[1]/255:.6f} {color[2]/255:.6f}\n")

def color_match_histogram(image: ImageType, usermap: ImageType) -> ImageType:
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
    #if cc == 4:
    #    image[..., 3] = alpha[..., 0]
    return image

def color_match_reinhard(image: ImageType, target: ImageType) -> ImageType:
    """
    Apply Reinhard color matching to an image based on a target image.
    Works only for BGR images and returns an BGR image.

    based on https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.

    Args:
        image (ImageType): The input image (BGR or BGRA or Grayscale).
        target (ImageType): The target image (BGR or BGRA or Grayscale).

    Returns:
        ImageType: The color-matched image in BGR format.
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

def color_mean(image: ImageType) -> ImageType:
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

def color_top_used(image: ImageType, top_n: int=8) -> List[tuple[int, int, int]]:
    """
    Find dominant colors in an image using k-means clustering.

    Args:
        image (np.ndarray): Input image in HxWxC format, assumed to be RGB.
        top_n (int): Number of top colors to return.

    Returns:
        List[tuple[int, int, int]]: List of top `top_n` colors.
    """
    if image.ndim < 3:
        image = np.expand_dims(image, axis=-1)

    if image.shape[2] != 3:
        image = image_convert(image, 3)

    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=int(top_n), n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    dominant_colors = np.round(dominant_colors).astype(int)
    sorted_colors = sorted(
        zip(dominant_colors, kmeans.labels_),
        key=lambda x: np.sum(kmeans.labels_ == x[1]),
        reverse=True
    )
    return [tuple(color) for color, _ in sorted_colors]

# ==============================================================================
# === COLOR ANALYSIS ===
# ==============================================================================

def rgb_to_hsv(bgr_color: PixelType) -> PixelType:
    return cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_RGB2HSV)[0, 0]

def hsv_to_rgb(hsl_color: PixelType) -> PixelType:
    return cv2.cvtColor(np.uint8([[hsl_color]]), cv2.COLOR_HSV2RGB)[0, 0]

def color_theory_complementary(color: PixelType) -> PixelType:
    color = rgb_to_hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    return hsv_to_rgb(color_a)

def color_theory_monochromatic(color: PixelType) -> tuple[PixelType, ...]:
    color = rgb_to_hsv(color)
    sat = 255 / 5
    val = 255 / 5
    color_a = pixel_hsv_adjust(color, 0, -1 * sat, -1 * val, mod_sat=True, mod_value=True)
    color_b = pixel_hsv_adjust(color, 0, -2 * sat, -2 * val, mod_sat=True, mod_value=True)
    color_c = pixel_hsv_adjust(color, 0, -3 * sat, -3 * val, mod_sat=True, mod_value=True)
    color_d = pixel_hsv_adjust(color, 0, -4 * sat, -4 * val, mod_sat=True, mod_value=True)
    return hsv_to_rgb(color_a), hsv_to_rgb(color_b), hsv_to_rgb(color_c), hsv_to_rgb(color_d)

def color_theory_split_complementary(color: PixelType) -> tuple[PixelType, ...]:
    color = rgb_to_hsv(color)
    color_a = pixel_hsv_adjust(color, 75, 0, 0)
    color_b = pixel_hsv_adjust(color, 105, 0, 0)
    return hsv_to_rgb(color_a), hsv_to_rgb(color_b)

def color_theory_analogous(color: PixelType) -> tuple[PixelType, ...]:
    color = rgb_to_hsv(color)
    color_a = pixel_hsv_adjust(color, 30, 0, 0)
    color_b = pixel_hsv_adjust(color, 15, 0, 0)
    color_c = pixel_hsv_adjust(color, 165, 0, 0)
    color_d = pixel_hsv_adjust(color, 150, 0, 0)
    return hsv_to_rgb(color_a), hsv_to_rgb(color_b), hsv_to_rgb(color_c), hsv_to_rgb(color_d)

def color_theory_triadic(color: PixelType) -> tuple[PixelType, ...]:
    color = rgb_to_hsv(color)
    color_a = pixel_hsv_adjust(color, 60, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    return hsv_to_rgb(color_a), hsv_to_rgb(color_b)

def color_theory_compound(color: PixelType) -> tuple[PixelType, ...]:
    color = rgb_to_hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    color_c = pixel_hsv_adjust(color, 150, 0, 0)
    return hsv_to_rgb(color_a), hsv_to_rgb(color_b), hsv_to_rgb(color_c)

def color_theory_square(color: PixelType) -> tuple[PixelType, ...]:
    color = rgb_to_hsv(color)
    color_a = pixel_hsv_adjust(color, 45, 0, 0)
    color_b = pixel_hsv_adjust(color, 90, 0, 0)
    color_c = pixel_hsv_adjust(color, 135, 0, 0)
    return hsv_to_rgb(color_a), hsv_to_rgb(color_b), hsv_to_rgb(color_c)

def color_theory_tetrad_custom(color: PixelType, delta:int=0) -> tuple[PixelType, ...]:
    color = rgb_to_hsv(color)

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
    return hsv_to_rgb(color_a), hsv_to_rgb(color_b), hsv_to_rgb(color_c), hsv_to_rgb(color_d)

def color_theory(image: ImageType, custom:int=0, scheme: EnumColorTheory=EnumColorTheory.COMPLIMENTARY) -> tuple[ImageType, ...]:

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

def image_gradient_expand(image: ImageType) -> None:
    image = image_convert(image, 3)
    image = cv2.resize(image, (256, 256))
    return image[0,:,:].reshape((256, 1, 3))

# Adapted from WAS Suite -- gradient_map
# https://github.com/WASasquatch/was-node-suite-comfyui
def image_gradient_map(image:ImageType, color_map:ImageType, reverse:bool=False) -> ImageType:
    if reverse:
        color_map = color_map[:,:,::-1]
    gray = image_grayscale(image)
    color_map = image_gradient_expand(color_map)
    return cv2.applyColorMap(gray, color_map)
