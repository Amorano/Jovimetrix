"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Composition Support
"""

from enum import Enum
from typing import Optional

import cv2
import numpy as np
from skimage import exposure
from blendmodes.blend import BlendType

from Jovimetrix import TYPE_IMAGE, TYPE_PIXEL
from Jovimetrix.sup.image import EnumMirrorMode, channel_count, pixel_bgr2hsv, pixel_hsv_adjust, \
    pixel_hsv2bgr, image_mirror
from Jovimetrix.sup.comp import comp_blend

# =============================================================================
# === ENUM GLOBALS ===
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

# =============================================================================
# === COLOR FUNCTIONS ===
# =============================================================================

def color_lut_from_image(image: TYPE_IMAGE, num_colors:int=256) -> TYPE_IMAGE:
    """Create X sized LUT from an RGB image."""
    image = cv2.resize(image, (num_colors, 1))
    return image.reshape(-1, 3).astype(np.uint8)

def color_match(image: TYPE_IMAGE, usermap: TYPE_IMAGE) -> TYPE_IMAGE:
    """Colorize one input based on the histogram matches."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    beta = cv2.cvtColor(usermap, cv2.COLOR_BGR2LAB)
    image = exposure.match_histograms(image, beta, channel_axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return comp_blend(usermap, image, blendOp=BlendType.LUMINOSITY)

def color_match_reinhard(image: TYPE_IMAGE, target: TYPE_IMAGE) -> TYPE_IMAGE:
    """Reinhard Color matching based on https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer."""
    lab_tar = cv2.cvtColor(target, cv2.COLOR_BGR2Lab)
    lab_ori = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    mean_tar, std_tar = cv2.meanStdDev(lab_tar)
    mean_ori, std_ori = cv2.meanStdDev(lab_ori)
    ratio = (std_tar/std_ori).reshape(-1)
    offset = (mean_tar - mean_ori*std_tar/std_ori).reshape(-1)
    lab_tar = cv2.convertScaleAbs(lab_ori*ratio + offset)
    return cv2.cvtColor(lab_tar, cv2.COLOR_Lab2BGR)

def color_match_custom_map(image: TYPE_IMAGE,
                   usermap: Optional[TYPE_IMAGE]=None,
                   colormap: int=cv2.COLORMAP_JET) -> TYPE_IMAGE:
    """Colorize one input based on custom GNU Octave/MATLAB map"""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image[:, :, 1]
    if usermap is not None:
        usermap = color_lut_from_image(usermap)
        return cv2.applyColorMap(image, usermap)
    return cv2.applyColorMap(image, colormap)

def color_match_heat_map(image: TYPE_IMAGE,
                  threshold:float=0.55,
                  colormap:int=cv2.COLORMAP_JET,
                  sigma:int=13) -> TYPE_IMAGE:
    """Colorize one input based on custom GNU Octave/MATLAB map"""

    threshold = min(1, max(0, threshold)) * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image[:, :, 1]
    image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

    sigma = max(3, sigma)
    if sigma % 2 == 0:
        sigma += 1
    sigmaY = sigma - 2

    image = cv2.GaussianBlur(image, (sigma, sigma), sigmaY)
    image = cv2.applyColorMap(image, colormap)
    return cv2.addWeighted(image, 0.5, image, 0.5, 0)

def color_mean(image: TYPE_IMAGE) -> TYPE_IMAGE:
    color = [0, 0, 0]
    if channel_count(image)[0] == 1:
        raw = int(np.mean(image))
        color = [raw] * 3
    else:
        # each channel....
        color = [
            int(np.mean(image[:,:,0])),
            int(np.mean(image[:,:,1])),
            int(np.mean(image[:,:,2])) ]
    return color

def color_theory_complementary(color: TYPE_PIXEL) -> TYPE_PIXEL:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    return pixel_hsv2bgr(color_a)

def color_theory_monochromatic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    sat = 255 / 5.
    val = 255 / 5.
    color_a = pixel_hsv_adjust(color, 0, -1 * sat, -1 * val, mod_sat=True, mod_value=True)
    color_b = pixel_hsv_adjust(color, 0, -2 * sat, -2 * val, mod_sat=True, mod_value=True)
    color_c = pixel_hsv_adjust(color, 0, -3 * sat, -3 * val, mod_sat=True, mod_value=True)
    color_d = pixel_hsv_adjust(color, 0, -4 * sat, -4 * val, mod_sat=True, mod_value=True)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c), pixel_hsv2bgr(color_d)

def color_theory_split_complementary(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 75, 0, 0)
    color_b = pixel_hsv_adjust(color, 105, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b)

def color_theory_analogous(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 30, 0, 0)
    color_b = pixel_hsv_adjust(color, 15, 0, 0)
    color_c = pixel_hsv_adjust(color, 165, 0, 0)
    color_d = pixel_hsv_adjust(color, 150, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c), pixel_hsv2bgr(color_d)

def color_theory_triadic(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 60, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b)

def color_theory_compound(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 90, 0, 0)
    color_b = pixel_hsv_adjust(color, 120, 0, 0)
    color_c = pixel_hsv_adjust(color, 150, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c)

def color_theory_square(color: TYPE_PIXEL) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)
    color_a = pixel_hsv_adjust(color, 45, 0, 0)
    color_b = pixel_hsv_adjust(color, 90, 0, 0)
    color_c = pixel_hsv_adjust(color, 135, 0, 0)
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c)

def color_theory_tetrad_custom(color: TYPE_PIXEL, delta:int=0) -> tuple[TYPE_PIXEL, TYPE_PIXEL, TYPE_PIXEL]:
    color = pixel_bgr2hsv(color)

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
    return pixel_hsv2bgr(color_a), pixel_hsv2bgr(color_b), pixel_hsv2bgr(color_c), pixel_hsv2bgr(color_d)

def color_theory(image: TYPE_IMAGE, custom:int=0, scheme: EnumColorTheory=EnumColorTheory.COMPLIMENTARY) -> tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE]:

    aR = aG = aB = bR = bG = bB = cR = cG = cB = dR = dG = dB = 0
    color = color_mean(image)
    match scheme:
        case EnumColorTheory.COMPLIMENTARY:
            a = color_theory_complementary(color)
            aB, aG, aR = a
        case EnumColorTheory.MONOCHROMATIC:
            a, b, c, d = color_theory_monochromatic(color)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
            dB, dG, dR = d
        case EnumColorTheory.SPLIT_COMPLIMENTARY:
            a, b = color_theory_split_complementary(color)
            aB, aG, aR = a
            bB, bG, bR = b
        case EnumColorTheory.ANALOGOUS:
            a, b, c, d = color_theory_analogous(color)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
            dB, dG, dR = d
        case EnumColorTheory.TRIADIC:
            a, b = color_theory_triadic(color)
            aB, aG, aR = a
            bB, bG, bR = b
        case EnumColorTheory.SQUARE:
            a, b, c = color_theory_square(color)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
        case EnumColorTheory.COMPOUND:
            a, b, c = color_theory_compound(color)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
        case EnumColorTheory.CUSTOM_TETRAD:
            a, b, c, d = color_theory_tetrad_custom(color, custom)
            aB, aG, aR = a
            bB, bG, bR = b
            cB, cG, cR = c
            dB, dG, dR = d

    h, w = image.shape[:2]

    return (
        np.full((h, w, 4), [aB, aG, aR, 255], dtype=np.uint8),
        np.full((h, w, 4), [bB, bG, bR, 255], dtype=np.uint8),
        np.full((h, w, 4), [cB, cG, cR, 255], dtype=np.uint8),
        np.full((h, w, 4), [dB, dG, dR, 255], dtype=np.uint8),
        np.full((h, w, 4), color + [255], dtype=np.uint8),
    )

# =============================================================================
# === TEST ===
# =============================================================================

if __name__ == "__main__":
    img2 = cv2.imread('./_res/img/test_mask.png', cv2.IMREAD_UNCHANGED)
    img = image_mirror(img2, EnumMirrorMode.YX)
    cv2.imwrite(f'./_res/tst/image-mirror1.png', img)
    img = image_mirror(img2, EnumMirrorMode.Y, reverse=True)
    cv2.imwrite(f'./_res/tst/image-mirror2.png', img)
    # testBlendModes()
    # testImageMerge()
