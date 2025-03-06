"""
Jovimetrix - Coordinates and Mapping
"""

from enum import Enum
from typing import Any, List, Tuple

import cv2
import numpy as np

from . import TAU, TYPE_IMAGE, TYPE_fCOORD2D, \
    image_convert, image_lerp, image_normalize

from .color import image_grayscale

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumProjection(Enum):
    NORMAL = 0
    POLAR = 5
    SPHERICAL = 10
    FISHEYE = 15
    PERSPECTIVE = 20

# ==============================================================================
# === IMAGE ===
# ==============================================================================

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

# ==============================================================================
# === COORDINATES ===
# ==============================================================================

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

# ==============================================================================
# === MAPPING ===
# ==============================================================================

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
