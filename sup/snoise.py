"""
// Copyright (c) 2008, Casey Duncan (casey dot duncan at gmail dot com)
// see LICENSE.txt for details
// $Id$
"""
import random
from math import floor
from typing import Optional, Sequence

import numpy as np

from vnoise._tables import GRAD3, PERM
from Jovimetrix.sup._tables import SIMPLEX, GRAD4, M_1_PI

"Native-code simplex noise functions"

# 2D simplex skew factors
F2 = 0.3660254037844386  # 0.5 * (sqrt(3.0) - 1.0)
G2 = 0.21132486540518713  # (3.0 - sqrt(3.0)) / 6.0


def _snoise2_impl(x: float, y: float) -> float:
    s = (x + y) * F2
    i = floor(x + s)
    j = floor(y + s)
    t = (i + j) * G2

    xx = [0.0, 0.0, 0.0]
    yy = [0.0, 0.0, 0.0]
    f = [0.0, 0.0, 0.0]

    noise = [0.0, 0.0, 0.0]
    g = [0, 0, 0]

    xx[0] = x - (i - t)
    yy[0] = y - (j - t)

    i1 = xx[0] > yy[0]
    j1 = xx[0] <= yy[0]

    xx[2] = xx[0] + G2 * 2.0 - 1.0
    yy[2] = yy[0] + G2 * 2.0 - 1.0
    xx[1] = xx[0] - i1 + G2
    yy[1] = yy[0] - j1 + G2

    I = int(i & 255)
    J = int(j & 255)
    g[0] = PERM[I + PERM[J]] % 12
    g[1] = PERM[I + i1 + PERM[J + j1]] % 12
    g[2] = PERM[I + 1 + PERM[J + 1]] % 12

    for c in range(0, 3):
        f[c] = 0.5 - xx[c] * xx[c] - yy[c] * yy[c]

    for c in range(0, 3):
        if f[c] > 0:
            noise[c] = (
                f[c] * f[c] * f[c] * f[c] * (GRAD3[g[c]][0] * xx[c] + GRAD3[g[c]][1] * yy[c])
            )

    return (noise[0] + noise[1] + noise[2]) * 70.0


def dot3(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def ASSIGN(a, v0, v1, v2):
    a[0] = v0
    a[1] = v1
    a[2] = v2


F3 = 1.0 / 3.0
G3 = 1.0 / 6.0


def _snoise3_impl(x: float, y: float, z: float) -> float:
    # int c, o1[3], o2[3], g[4], I, J, K;
    o1 = [0, 0, 0]
    o2 = [0, 0, 0]
    g = [0, 0, 0, 0]

    f = [0.0] * 4
    noise = [0.0] * 4
    s = (x + y + z) * F3
    i = np.floor(x + s)
    j = np.floor(y + s)
    k = np.floor(z + s)
    t = (i + j + k) * G3

    pos = np.zeros((4, 3), dtype="float")
    # float pos[4][3];

    pos[0][0] = x - (i - t)
    pos[0][1] = y - (j - t)
    pos[0][2] = z - (k - t)

    if pos[0][0] >= pos[0][1]:
        if pos[0][1] >= pos[0][2]:
            ASSIGN(o1, 1, 0, 0)
            ASSIGN(o2, 1, 1, 0)
        elif pos[0][0] >= pos[0][2]:
            ASSIGN(o1, 1, 0, 0)
            ASSIGN(o2, 1, 0, 1)
        else:
            ASSIGN(o1, 0, 0, 1)
            ASSIGN(o2, 1, 0, 1)
    else:
        if pos[0][1] < pos[0][2]:
            ASSIGN(o1, 0, 0, 1)
            ASSIGN(o2, 0, 1, 1)
        elif pos[0][0] < pos[0][2]:
            ASSIGN(o1, 0, 1, 0)
            ASSIGN(o2, 0, 1, 1)
        else:
            ASSIGN(o1, 0, 1, 0)
            ASSIGN(o2, 1, 1, 0)

    for c in range(0, 3):
        pos[3][c] = pos[0][c] - 1.0 + 3.0 * G3
        pos[2][c] = pos[0][c] - o2[c] + 2.0 * G3
        pos[1][c] = pos[0][c] - o1[c] + G3

    I = int(i & 255)
    J = int(j & 255)
    K = int(k & 255)
    g[0] = PERM[I + PERM[J + PERM[K]]] % 12
    g[1] = PERM[I + o1[0] + PERM[J + o1[1] + PERM[o1[2] + K]]] % 12
    g[2] = PERM[I + o2[0] + PERM[J + o2[1] + PERM[o2[2] + K]]] % 12
    g[3] = PERM[I + 1 + PERM[J + 1 + PERM[K + 1]]] % 12

    for c in range(0, 4):
        f[c] = 0.6 - pos[c][0] * pos[c][0] - pos[c][1] * pos[c][1] - pos[c][2] * pos[c][2]

    for c in range(0, 4):
        if f[c] > 0:
            noise[c] = f[c] * f[c] * f[c] * f[c] * dot3(pos[c], GRAD3[g[c]])

    return (noise[0] + noise[1] + noise[2] + noise[3]) * 32.0


def _fbm_noise3_impl(
    x: float, y: float, z: float, octaves: int, persistence: float, lacunarity: float
) -> float:
    freq = 1.0
    amp = 1.0
    max = 1.0
    total = _snoise3_impl(x, y, z)
    for i in range(1, octaves):
        freq *= lacunarity
        amp *= persistence
        max += amp
        total += _snoise3_impl(x * freq, y * freq, z * freq) * amp
    return total / max


def _dot4(v1, x, y, z, w):
    return v1[0] * x + v1[1] * y + v1[2] * z + v1[3] * w


F4 = 0.30901699437494745  # /* (sqrt(5.0) - 1.0) / 4.0 */
G4 = 0.1381966011250105  # /* (5.0 - sqrt(5.0)) / 20.0 */


def _noise4_impl(x: float, y: float, z: float, w: float) -> float:
    noise = [0.0] * 5

    s = (x + y + z + w) * F4
    i = np.floor(x + s)
    j = np.floor(y + s)
    k = np.floor(z + s)
    l = np.floor(w + s)
    t = (i + j + k + l) * G4

    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)
    w0 = w - (l - t)

    c = (
        (x0 > y0) * 32
        + (x0 > z0) * 16
        + (y0 > z0) * 8
        + (x0 > w0) * 4
        + (y0 > w0) * 2
        + (z0 > w0)
    )
    i1 = SIMPLEX[c][0] >= 3
    j1 = SIMPLEX[c][1] >= 3
    k1 = SIMPLEX[c][2] >= 3
    l1 = SIMPLEX[c][3] >= 3
    i2 = SIMPLEX[c][0] >= 2
    j2 = SIMPLEX[c][1] >= 2
    k2 = SIMPLEX[c][2] >= 2
    l2 = SIMPLEX[c][3] >= 2
    i3 = SIMPLEX[c][0] >= 1
    j3 = SIMPLEX[c][1] >= 1
    k3 = SIMPLEX[c][2] >= 1
    l3 = SIMPLEX[c][3] >= 1

    x1 = x0 - i1 + G4
    y1 = y0 - j1 + G4
    z1 = z0 - k1 + G4
    w1 = w0 - l1 + G4
    x2 = x0 - i2 + 2.0 * G4
    y2 = y0 - j2 + 2.0 * G4
    z2 = z0 - k2 + 2.0 * G4
    w2 = w0 - l2 + 2.0 * G4
    x3 = x0 - i3 + 3.0 * G4
    y3 = y0 - j3 + 3.0 * G4
    z3 = z0 - k3 + 3.0 * G4
    w3 = w0 - l3 + 3.0 * G4
    x4 = x0 - 1.0 + 4.0 * G4
    y4 = y0 - 1.0 + 4.0 * G4
    z4 = z0 - 1.0 + 4.0 * G4
    w4 = w0 - 1.0 + 4.0 * G4

    I = int(i & 255)
    J = int(j & 255)
    K = int(k & 255)
    L = int(l & 255)
    gi0 = PERM[I + PERM[J + PERM[K + PERM[L]]]] & 0x1F
    gi1 = PERM[I + i1 + PERM[J + j1 + PERM[K + k1 + PERM[L + l1]]]] & 0x1F
    gi2 = PERM[I + i2 + PERM[J + j2 + PERM[K + k2 + PERM[L + l2]]]] & 0x1F
    gi3 = PERM[I + i3 + PERM[J + j3 + PERM[K + k3 + PERM[L + l3]]]] & 0x1F
    gi4 = PERM[I + 1 + PERM[J + 1 + PERM[K + 1 + PERM[L + 1]]]] & 0x1F
    # float t0, t1, t2, t3, t4;

    t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0
    if t0 >= 0.0:
        t0 *= t0
        noise[0] = t0 * t0 * _dot4(GRAD4[gi0], x0, y0, z0, w0)

    t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1
    if t1 >= 0.0:
        t1 *= t1
        noise[1] = t1 * t1 * _dot4(GRAD4[gi1], x1, y1, z1, w1)

    t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2
    if t2 >= 0.0:
        t2 *= t2
        noise[2] = t2 * t2 * _dot4(GRAD4[gi2], x2, y2, z2, w2)

    t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3
    if t3 >= 0.0:
        t3 *= t3
        noise[3] = t3 * t3 * _dot4(GRAD4[gi3], x3, y3, z3, w3)

    t4 = 0.6 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4
    if t4 >= 0.0:
        t4 *= t4
        noise[4] = t4 * t4 * _dot4(GRAD4[gi4], x4, y4, z4, w4)

    return 27.0 * (noise[0] + noise[1] + noise[2] + noise[3] + noise[4])


def _fbm_noise4_impl(
    x: float, y: float, z: float, w: float, octaves: int, persistence: float, lacunarity: float
) -> float:
    freq = 1.0
    amp = 1.0
    max = 1.0
    total = _noise4_impl(x, y, z, w)

    for i in range(1, octaves):
        freq *= lacunarity
        amp *= persistence
        max += amp
        total += _noise4_impl(x * freq, y * freq, z * freq, w * freq) * amp
    return total / max


class SNoise:
    def __init__(self, seed: Optional[int] = None):

        if seed is not None:
            self.seed(seed)
        else:
            self._set_perm(PERM)

    def seed(self, s: int) -> None:
        perm = list(PERM)
        random.Random(s).shuffle(perm)
        self._set_perm(perm)

    def _set_perm(self, perm: Sequence[int]) -> None:
        self._perm = np.array(list(perm) * 2, dtype=np.uint8)

    def noise2(
        self,
        x: float,
        y: float,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeatx: Optional[int] = None,
        repeaty: Optional[int] = None,
        base: int = 0,
    ) -> float:
        """
        noise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=None, repeaty=None, base=0.0)
        return simplex noise value for specified 2D coordinate.

        octaves -- specifies the number of passes, defaults to 1 (simple noise).

        persistence -- specifies the amplitude of each successive octave relative
        to the one below it. Defaults to 0.5 (each higher octave's amplitude
        is halved). Note the amplitude of the first pass is always 1.0.

        lacunarity -- specifies the frequency of each successive octave relative
        "to the one below it, similar to persistence. Defaults to 2.0.

        repeatx, repeaty -- specifies the interval along each axis when
        "the noise values repeat. This can be used as the tile size for creating
        "tileable textures

        base -- specifies a fixed offset for the noise coordinates. Useful for
        generating different noise textures with the same repeat interval
        """
        z = 0.0

        if octaves <= 0:
            raise ValueError("Expected octaves value > 0")

        if repeatx is None and repeaty is None:
            # Flat noise, no tiling
            freq = 1.0
            amp = 1.0
            max = 1.0
            total = _snoise2_impl(x + z, y + z)

            for i in range(1, octaves):
                freq *= lacunarity
                amp *= persistence
                max += amp
                total += _snoise2_impl(x * freq + z, y * freq + z) * amp

            return total / max
        else:  # Tiled noise
            w = z
            if repeaty is not None:
                yf = y * 2.0 / repeaty
                yr = repeaty * M_1_PI * 0.5
                vy = np.sin(yf)  # originally fast_sin
                vyz = np.cos(yf)  # originally fast_cos
                y = vy * yr
                w += vyz * yr
                if repeatx is None:
                    return _fbm_noise3_impl(x, y, w, octaves, persistence, lacunarity)
            if repeatx is not None:
                xf = x * 2.0 / repeatx
                xr = repeatx * M_1_PI * 0.5
                vx = np.sin(xf)
                vxz = np.cos(xf)
                x = vx * xr
                z += vxz * xr
                if repeaty is None:
                    return _fbm_noise3_impl(x, y, z, octaves, persistence, lacunarity)
            return _fbm_noise4_impl(x, y, z, w, octaves, persistence, lacunarity)

    def noise3(
        self,
        x: float,
        y: float,
        z: float,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> float:
        """
        noise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0) return simplex noise value for
        specified 3D coordinate

        octaves -- specifies the number of passes, defaults to 1 (simple noise).

        persistence -- specifies the amplitude of each successive octave relative
        to the one below it. Defaults to 0.5 (each higher octave's amplitude
        is halved). Note the amplitude of the first pass is always 1.0.

        lacunarity -- specifies the frequency of each successive octave relative
        to the one below it, similar to persistence. Defaults to 2.0.
        """
        if octaves == 1:
            # Single octave, return simple noise
            return _snoise3_impl(x, y, z)
        elif octaves > 1:
            return _fbm_noise3_impl(x, y, z, octaves, persistence, lacunarity)
        else:
            raise ValueError("Expected octaves value > 0")

    def noise4(
        self,
        x: float,
        y: float,
        z: float,
        w: float,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> float:
        """
        noise4(x, y, z, w, octaves=1, persistence=0.5, lacunarity=2.0) return simplex noise value for
        specified 4D coordinate

        octaves -- specifies the number of passes, defaults to 1 (simple noise).

        persistence -- specifies the amplitude of each successive octave relative
        to the one below it. Defaults to 0.5 (each higher octave's amplitude
        is halved). Note the amplitude of the first pass is always 1.0.

        lacunarity -- specifies the frequency of each successive octave relative
        to the one below it, similar to persistence. Defaults to 2.0.
        """
        if octaves == 1:
            # Single octave, return simple noise
            return _noise4_impl(x, y, z, w)
        elif octaves > 1:
            return _fbm_noise4_impl(x, y, z, w, octaves, persistence, lacunarity)
        else:
            raise ValueError("Expected octaves value > 0")