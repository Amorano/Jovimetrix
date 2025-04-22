""" Jovimetrix - Coordinates and Mapping """

from enum import Enum
from typing import List

import cv2
import numpy as np

from cozy_comfyui.image import \
    TAU, \
    Coord2D_Float, ImageType

from cozy_comfyui.image.convert import \
    ImageType

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
# === COORDINATES ===
# ==============================================================================

def coord_cart2polar(x: float, y: float) -> Coord2D_Float:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def coord_polar2cart(r: float, theta: float) -> Coord2D_Float:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def coord_default(width:int, height:int, origin:Coord2D_Float=None) -> Coord2D_Float:
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

def coord_fisheye(width: int, height: int, distortion: float) -> tuple[ImageType, ImageType]:
    map_x, map_y = np.meshgrid(np.linspace(0., 1., width), np.linspace(0., 1., height))
    # normalized
    xnd, ynd = (2 * map_x - 1), (2 * map_y - 1)
    rd = np.sqrt(xnd**2 + ynd**2)
    # fish-eye distortion
    condition = (dist := 1 - distortion * (rd**2)) == 0
    xdu, ydu = np.where(condition, xnd, xnd / dist), np.where(condition, ynd, ynd / dist)
    xu, yu = ((xdu + 1) * width) / 2, ((ydu + 1) * height) / 2
    return xu.astype(np.float32), yu.astype(np.float32)

def coord_perspective(width: int, height: int, pts: List[Coord2D_Float]) -> ImageType:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0], pts[:, 1]])
    return cv2.getPerspectiveTransform(object_pts, pts)

def coord_sphere(width: int, height: int, radius: float) -> tuple[ImageType, ImageType]:
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

def remap_fisheye(image: ImageType, distort: float) -> ImageType:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    map_x, map_y = coord_fisheye(width, height, distort)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #if cc == 1:
    #    image = image[..., 0]
    return image

def remap_perspective(image: ImageType, pts: list) -> ImageType:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pts = coord_perspective(width, height, pts)
    image = cv2.warpPerspective(image, pts, (width, height))
    #if cc == 1:
    #    image = image[..., 0]
    return image

def remap_polar(image: ImageType) -> ImageType:
    """Re-projects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    h, w = image.shape[:2]
    radius = max(w, h)
    return cv2.linearPolar(image, (h // 2, w // 2), radius // 2, cv2.WARP_INVERSE_MAP)

def remap_sphere(image: ImageType, radius: float) -> ImageType:
    height, width = image.shape[:2]
    map_x, map_y = coord_sphere(width, height, radius)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
