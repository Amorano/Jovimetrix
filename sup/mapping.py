"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Mapping Support
"""

import math
from enum import Enum
from typing import Any

import cv2
import numpy as np
import scipy as sp
from loguru import logger

from Jovimetrix import TYPE_IMAGE, TYPE_COORD

HALFPI = math.pi / 2
TAU = math.pi * 2

# =============================================================================
# === ENUM GLOBALS ===
# =============================================================================

class EnumProjection(Enum):
    NORMAL = 0
    POLAR = 5
    SPHERICAL = 10
    FISHEYE = 15

# =============================================================================

def cart2polar(x, y) -> tuple[Any, Any]:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta) -> tuple:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# =============================================================================

def coord_sphere(width: int, height: int, radius: float) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    theta, phi = np.meshgrid(np.linspace(0, TAU, width), np.linspace(0, np.pi, height))
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    # z = radius * np.cos(phi)
    x_image = (x + 1) * (width - 1) / 2
    y_image = (y + 1) * (height - 1) / 2
    return x_image.astype(np.float32), y_image.astype(np.float32)

def coord_polar(data, origin=None) -> tuple:
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def coord_perspective(width: int, height: int, pts: list[TYPE_COORD]) -> TYPE_IMAGE:
    object_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts = np.float32(pts)
    pts = np.column_stack([pts[:, 0], pts[:, 1]])
    return cv2.getPerspectiveTransform(object_pts, pts)

def coord_fisheye(width: int, height: int, distortion: float) -> tuple[TYPE_IMAGE, TYPE_IMAGE]:
    map_x, map_y = np.meshgrid(np.linspace(0., 1., width), np.linspace(0., 1., height))
    # normalized
    xnd, ynd = (2 * map_x - 1), (2 * map_y - 1)
    rd = np.sqrt(xnd**2 + ynd**2)
    # fish-eye distortion
    condition = (dist := 1 - distortion * (rd**2)) == 0
    xdu, ydu = np.where(condition, xnd, xnd / dist), np.where(condition, ynd, ynd / dist)
    xu, yu = ((xdu + 1) * width) / 2, ((ydu + 1) * height) / 2
    return xu.astype(np.float32), yu.astype(np.float32)

def remap_sphere(image: TYPE_IMAGE, radius: float) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    map_x, map_y = coord_sphere(width, height, radius)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_polar(image: TYPE_IMAGE) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    map_x, map_y = coord_polar(width, height)
    map_x = map_x * width
    map_y = map_y * height
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def remap_polar(image: TYPE_IMAGE, origin=None) -> tuple[np.ndarray, Any, Any]:
    """Re-projects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = image.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = coord_polar(image, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0]
    yi += origin[1]
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi))
    bands = []
    for band in image.T:
        zi = sp.ndimage.map_coordinates(band, coords, order=1)
        bands.append(zi.reshape((nx, ny)))
    return np.dstack(bands)
    # return output, r_i, theta_i

def remap_perspective(image: TYPE_IMAGE, pts: list) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    pts = coord_perspective(width, height, pts)
    return cv2.warpPerspective(image, pts, (width, height))

def remap_fisheye(image: TYPE_IMAGE, distort: float) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    map_x, map_y = coord_fisheye(width, height, distort)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# =============================================================================
# === ZE MAIN ===
# =============================================================================

def testTRS() -> None:
    image = cv2.imread("./_res/img/test_alpha.png")
    h, w = image.shape[:2]
    pts = [
        [0.1 * w, 0.1 * h],
        [0.7 * w, 0.3 * h],
        [0.9 * w, 0.9 * h],
        [0.1 * w, 0.9 * h]
    ]
    remap = [
        ('perspective', remap_perspective(image, pts)),
        ('fisheye', remap_fisheye(image, 2)),
        ('sphere', remap_sphere(image, 0.1)),
        ('sphere', remap_sphere(image, 0.5)),
        ('sphere', remap_sphere(image, 1)),
        ('sphere', remap_sphere(image, 2)),
        ('polar', remap_polar(image)),
    ]
    idx_remap = 0
    while True:
        title, image,  = remap[idx_remap]
        cv2.imshow("", image)
        logger.debug(title)
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        idx_remap = (idx_remap + 1) % len(remap)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    testTRS()
