"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animation Support
"""

import math
import inspect
from enum import Enum

import numpy as np
from numba import jit
from loguru import logger

__all__ = ["Ease", "Wave"]

HALFPI = math.pi / 2
TAU = math.pi * 2

# =============================================================================
# === EXCEPTIONAL ===
# =============================================================================

class BadOperatorException(Exception):
    """Exception for bad operators."""
    pass

TYPE_NUMBER = int|float|np.ndarray
MODULE = inspect.getmodule(inspect.currentframe())

# =============================================================================
# === EASING ===
# =============================================================================

class EnumEase(Enum):
    QUAD_IN = 10
    QUAD_OUT = 11
    QUAD_IN_OUT = 12

    CUBIC_IN = 20
    CUBIC_OUT = 21
    CUBIC_IN_OUT = 22

    QUARTIC_IN = 30
    QUARTIC_OUT = 31
    QUARTIC_IN_OUT = 32

    QUINTIC_IN = 40
    QUINTIC_OUT = 41
    QUINTIC_IN_OUT = 42

    SIN_IN = 50
    SIN_OUT = 51
    SIN_IN_OUT = 52

    CIRCULAR_IN = 60
    CIRCULAR_OUT = 61
    CIRCULAR_IN_OUT = 62

    EXPONENTIAL_IN = 70
    EXPONENTIAL_OUT = 71
    EXPONENTIAL_IN_OUT = 72

    ELASTIC_IN = 80
    ELASTIC_OUT = 81
    ELASTIC_IN_OUT = 82

    BACK_IN = 90
    BACK_OUT = 91
    BACK_IN_OUT = 92

    BOUNCE_IN = 100
    BOUNCE_OUT = 101
    BOUNCE_IN_OUT = 102

@jit(parallel=True, cache=True)
def ease_quad_in(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return t * t

@jit(parallel=True, cache=True)
def ease_quad_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return -(t * (t - 2))

@jit(parallel=True, cache=True)
def ease_quad_in_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return np.where(t < 0.5, 2 * t * t, (-2 * t * t) + (4 * t) - 1)

@jit(parallel=True, cache=True)
def ease_cubic_in(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return t * t * t

@jit(parallel=True, cache=True)
def ease_cubic_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return (t - 1) * (t - 1) * (t - 1) + 1

@jit(parallel=True, cache=True)
def ease_cubic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 4 * t * t * t,
                    0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

@jit(parallel=True, cache=True)
def ease_quartic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t * t

@jit(parallel=True, cache=True)
def ease_quartic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) * (1 - t) + 1

@jit(parallel=True, cache=True)
def ease_quartic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 8 * t * t * t * t,
                    -8 * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1)

@jit(parallel=True, cache=True)
def ease_quintic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t * t * t

@jit(parallel=True, cache=True)
def ease_quintic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1

@jit(parallel=True, cache=True)
def ease_quintic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 16 * t * t * t * t * t,
                    0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

@jit(parallel=True, cache=True)
def ease_sin_in(t: np.ndarray) -> np.ndarray:
    return np.sin((t - 1) * HALFPI) + 1

@jit(parallel=True, cache=True)
def ease_sin_out(t: np.ndarray) -> np.ndarray:
    return np.sin(t * HALFPI)

@jit(parallel=True, cache=True)
def ease_sin_in_out(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1 - np.cos(t * math.pi))

@jit(parallel=True, cache=True)
def ease_circular_in(t: np.ndarray) -> np.ndarray:
    return 1 - np.sqrt(1 - (t * t))

@jit(parallel=True, cache=True)
def ease_circular_out(t: np.ndarray) -> np.ndarray:
    return np.sqrt((2 - t) * t)

@jit(parallel=True, cache=True)
def ease_circular_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * (1 - np.sqrt(1 - 4 * (t * t))),
                    0.5 * (np.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1))

@jit(parallel=True, cache=True)
def ease_exponential_in(t: np.ndarray) -> np.ndarray:
    return np.where(t == 0, 0, np.power(2, 10 * (t - 1)))

@jit(parallel=True, cache=True)
def ease_exponential_out(t: np.ndarray) -> np.ndarray:
    return np.where(t == 1, 1, 1 - np.power(2, -10 * t))

@jit(parallel=True, cache=True)
def ease_exponential_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t == 0, t, np.where(t < 0.5, 0.5 * np.power(2, (20 * t) - 10),
                                        -0.5 * np.power(2, (-20 * t) + 10) + 1))

@jit(parallel=True, cache=True)
def ease_elastic_in(t: np.ndarray) -> np.ndarray:
    return np.sin(13 * HALFPI * t) * np.power(2, 10 * (t - 1))

@jit(parallel=True, cache=True)
def ease_elastic_out(t: np.ndarray) -> np.ndarray:
    return np.sin(-13 * HALFPI * (t + 1)) * np.power(2, -10 * t) + 1

@jit(parallel=True, cache=True)
def ease_elastic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * np.sin(13 * HALFPI * (2 * t)) * np.power(2, 10 * ((2 * t) - 1)),
                    0.5 * (np.sin(-13 * HALFPI * ((2 * t - 1) + 1)) * np.power(2, -10 * (2 * t - 1)) + 2))

@jit(parallel=True, cache=True)
def ease_back_in(t: np.ndarray) -> np.ndarray:
    return t * t * t - t * np.sin(t * math.pi)

@jit(parallel=True, cache=True)
def ease_back_out(t: np.ndarray) -> np.ndarray:
    p = 1 - t
    return 1 - (p * p * p - p * np.sin(p * math.pi))

@jit(parallel=True, cache=True)
def ease_back_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * (2 * t) * (2 * t) * (2 * t) - (2 * t) * np.sin((2 * t) * math.pi),
                    0.5 * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) - (1 - (2 * t - 1)) * np.sin((1 - (2 * t - 1)) * math.pi) + 0.5)

@jit(parallel=True, cache=True)
def ease_bounce_in(t: np.ndarray) -> np.ndarray:
    return 1 - ease_bounce_out(1 - t)

@jit(parallel=True, cache=True)
def ease_bounce_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 4 / 11, 121 * t * t / 16,
        np.where(t < 8 / 11, (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0,
        np.where(t < 9 / 10, (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0,
                (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0)))

@jit(parallel=True, cache=True)
def ease_bounce_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * ease_bounce_in(t * 2), 0.5 * ease_bounce_out(t * 2 - 1) + 0.5)

def ease_op(op: EnumEase,
            start: float=0, end: float=1, duration: float=1,
            alpha: float=1., clip: tuple[int, int]=(0, 1)) -> np.ndarray:
    """
    Compute eased values.

    Parameters:
        op (EaseOP): Easing operator.
        start (float): Starting value.
        end (float): Ending value.
        duration (float): Duration of the easing.
        alpha (float): Alpha values.
        clip (tuple[int, int]): Clip range.

    Returns:
        TYPE_NUMBER: Eased value(s)
    """
    if (func := getattr(MODULE, f"ease_{op.lower()}", None)) is None:
        raise BadOperatorException(str(op))
    t = clip[0] * (1 - alpha) + clip[1] * alpha
    duration = max(min(duration, 1), 0)
    t /= duration
    a = func(t)
    return end * a + start * (1 - a)

# =============================================================================
# === WAVE FUNCTIONS SIMPLE ===
# =============================================================================

class EnumWave(Enum):
    SIN = 0
    SIN_INV = 1
    SIN_ABS = 2
    COS = 3
    COS_INV = 4
    COS_ABS = 5
    SAWTOOTH = 6
    TRIANGLE = 7
    SQUARE = 8
    PULSE = 9
    RAMP = 10
    STEP = 11
    EXPONENTIAL = 12
    LOGARITHMIC = 13
    NOISE = 14
    HAVERSINE = 15
    RECTANGULAR_PULSE = 16
    GAUSSIAN = 17
    CHIRP = 18

@jit(parallel=True, cache=True)
def wave_sin(phase: float, frequency: float, amplitude: float, offset: float,
             timestep: float) -> float:
    return amplitude * np.sin(frequency * TAU * timestep + phase) + offset

@jit(parallel=True, cache=True)
def wave_sin_inv(phase: float, frequency: float, amplitude: float, offset: float,
                 timestep: float) -> float:
    return -amplitude * np.sin(frequency * TAU * timestep + phase) + offset

@jit(parallel=True, cache=True)
def wave_sin_abs(phase: float, frequency: float, amplitude: float, offset: float,
                 timestep: float) -> float:
    return np.abs(amplitude * np.sin(frequency * TAU * timestep + phase)) + offset

@jit(parallel=True, cache=True)
def wave_cos(phase: float, frequency: float, amplitude: float, offset: float,
             timestep: float) -> float:
    return amplitude * np.cos(frequency * TAU * timestep + phase) + offset

@jit(parallel=True, cache=True)
def wave_cos_inv(phase: float, frequency: float, amplitude: float, offset: float,
                 timestep: float) -> float:
    return -amplitude * np.cos(frequency * TAU * timestep + phase) + offset

@jit(parallel=True, cache=True)
def wave_cos_abs(phase: float, frequency: float, amplitude: float, offset: float,
                 timestep: float) -> float:
    return np.abs(amplitude * np.cos(frequency * TAU * timestep + phase)) + offset

@jit(parallel=True, cache=True)
def wave_sawtooth(phase: float, frequency: float, amplitude: float, offset: float,
                  timestep: float) -> float:
    return amplitude * (2 * (frequency * timestep + phase) % 1 - 0.5) + offset

@jit(parallel=True, cache=True)
def wave_triangle(phase: float, frequency: float, amplitude: float, offset: float,
                  timestep: float) -> float:
    return amplitude * (4 * np.abs((frequency * timestep + phase) % 1 - 0.5) - 1) + offset

@jit(parallel=True, cache=True)
def wave_ramp(phase: float, frequency: float, amplitude: float, offset: float,
              timestep: float) -> float:
    return amplitude * (frequency * timestep + phase % 1) + offset

@jit(parallel=True, cache=True)
def wave_step(phase: float, frequency: float, amplitude: float, offset: float,
              timestep: float) -> float:
    return amplitude * np.heaviside(frequency * timestep + phase, 1) + offset

@jit(parallel=True, cache=True)
def wave_haversine(phase: float, frequency: float, amplitude: float, offset: float,
                   timestep: float) -> float:
    return amplitude * (1 - np.cos(frequency * TAU * (timestep + phase))) + offset

@jit(parallel=True, cache=True)
def wave_noise(phase: float, frequency: float, amplitude: float, offset: float,
               timestep: float) -> float:
    return amplitude * np.random.uniform(-1, 1) + offset

# =============================================================================
# === WAVE FUNCTIONS COMPLEX ===
# =============================================================================

@jit(parallel=True, cache=True)
def wave_square(phase: float, frequency: float, amplitude: float, offset: float,
                timestep: float) -> float:
    return amplitude * np.sign(np.sin(TAU * timestep + phase) - frequency) + offset

@jit(parallel=True, cache=True)
def wave_pulse(phase: float, frequency: float, amplitude: float, offset: float,
               timestep: float) -> float:
    return amplitude * np.sign(np.sin(TAU * timestep + phase) - frequency) + offset

@jit(parallel=True, cache=True)
def wave_exponential(phase: float, frequency: float, amplitude: float,
                     offset: float, timestep: float) -> float:
    return amplitude * np.exp(-frequency * (timestep + phase)) + offset

@jit(parallel=True, cache=True)
def wave_rectangular_pulse(phase: float, frequency: float, amplitude: float,
                           offset: float, timestep: float) -> float:
    return amplitude * np.heaviside(timestep + phase, 1) * np.heaviside(-(timestep + phase) + frequency, 1) + offset

@jit(parallel=True, cache=True)
def wave_logarithmic(phase: float, frequency: float, amplitude: float, offset: float,
                     timestep: float) -> float:
    return amplitude * np.log10(timestep + phase) / np.log10(frequency) + offset

@jit(parallel=True, cache=True)
def wave_chirp(phase: float, frequency: float, amplitude: float, offset: float,
               timestep: float) -> float:
    return amplitude * np.sin(TAU * frequency * (timestep + phase)**2) + offset

####

@jit(parallel=True, cache=True)
def wave_gaussian(phase: float, mean: float, amplitude: float, offset: float,
                  timestep: float, std_dev: float = 1) -> float:
    return amplitude * np.exp(-0.5 * ((timestep + phase - mean) / std_dev)**2) + offset

def wave_op(op: EnumEase, phase: float, frequency: float, amplitude: float,
            offset: float, timestep: float, std_dev: float=1) -> np.ndarray:

    op = op.lower()
    if (func := getattr(MODULE, f"wave_{op}", None)) is None:
        raise BadOperatorException(str(op))
    """
    phase = float(phase)
    frequency = float(frequency)
    amplitude = float(amplitude)
    offset = float(offset)
    timestep = float(timestep)
    std_dev = float(std_dev)
    """
    if op.endswith('gaussian'):
        return func(phase, frequency, amplitude, offset, timestep, std_dev)
    return func(phase, frequency, amplitude, offset, timestep)
