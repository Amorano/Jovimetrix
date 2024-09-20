"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Noise Support
"""

from enum import Enum

# ==============================================================================

class EnumNoise(Enum):
    PERLIN_1D = 10
    PERLIN_2D = 20
    PERLIN_2D_RGB = 30
    PERLIN_2D_RGBA = 40

class EnumNoise2(Enum):
    OPENSIMPLEX2 = 10
    OPENSIMPLEX2S = 20
    CELLULAR = 30
    PERLIN = 40
    VALUECUBIC = 50
    VALUE = 60

class EnumFractal(Enum):
    NONE = 10
    FBM = 20
    RIGID = 30
    PINGPONG = 40
    WARP = 60
    WARP_PROGRESSIVE = 50
