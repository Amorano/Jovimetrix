"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animation Support
"""

import inspect
from enum import Enum
from typing import Tuple, Union

import numpy as np
from numba import jit
from loguru import logger

__all__ = ["Ease", "Wave"]

TYPE_NUMBER = Union[int|float|np.ndarray]

# =============================================================================
# === EXCEPTIONAL ===
# =============================================================================

class BadOperatorException(Exception):
    """Exception for bad operators."""
    pass

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

@jit(cache=True)
def ease_quad_in(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return t * t

@jit(cache=True)
def ease_quad_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return -(t * (t - 2))

@jit(cache=True)
def ease_quad_in_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return np.where(t < 0.5, 2 * t * t, (-2 * t * t) + (4 * t) - 1)

@jit(cache=True)
def ease_cubic_in(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return t * t * t

@jit(cache=True)
def ease_cubic_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return (t - 1) * (t - 1) * (t - 1) + 1

@jit(cache=True)
def ease_cubic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 4 * t * t * t,
                    0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

@jit(cache=True)
def ease_quartic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t * t

@jit(cache=True)
def ease_quartic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) * (1 - t) + 1

@jit(cache=True)
def ease_quartic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 8 * t * t * t * t,
                    -8 * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1)

@jit(cache=True)
def ease_quintic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t * t * t

@jit(cache=True)
def ease_quintic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1

@jit(cache=True)
def ease_quintic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 16 * t * t * t * t * t,
                    0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

@jit(cache=True)
def ease_sin_in(t: np.ndarray) -> np.ndarray:
    return np.sin((t - 1) * np.pi * 0.5) + 1

@jit(cache=True)
def ease_sin_out(t: np.ndarray) -> np.ndarray:
    return np.sin(t * np.pi * 0.5)

@jit(cache=True)
def ease_sin_in_out(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1 - np.cos(t * np.pi))

@jit(cache=True)
def ease_circular_in(t: np.ndarray) -> np.ndarray:
    return 1 - np.sqrt(1 - (t * t))

@jit(cache=True)
def ease_circular_out(t: np.ndarray) -> np.ndarray:
    return np.sqrt((2 - t) * t)

@jit(cache=True)
def ease_circular_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * (1 - np.sqrt(1 - 4 * (t * t))),
                    0.5 * (np.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1))

@jit(cache=True)
def ease_exponential_in(t: np.ndarray) -> np.ndarray:
    return np.where(t == 0, 0, np.power(2, 10 * (t - 1)))

@jit(cache=True)
def ease_exponential_out(t: np.ndarray) -> np.ndarray:
    return np.where(t == 1, 1, 1 - np.power(2, -10 * t))

@jit(cache=True)
def ease_exponential_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t == 0, t, np.where(t < 0.5, 0.5 * np.power(2, (20 * t) - 10),
                                        -0.5 * np.power(2, (-20 * t) + 10) + 1))

@jit(cache=True)
def ease_elastic_in(t: np.ndarray) -> np.ndarray:
    return np.sin(13 * np.pi * 0.5 * t) * np.power(2, 10 * (t - 1))

@jit(cache=True)
def ease_elastic_out(t: np.ndarray) -> np.ndarray:
    return np.sin(-13 * np.pi * 0.5 * (t + 1)) * np.power(2, -10 * t) + 1

@jit(cache=True)
def ease_elastic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * np.sin(13 * np.pi * 0.5 * (2 * t)) * np.power(2, 10 * ((2 * t) - 1)),
                    0.5 * (np.sin(-13 * np.pi * 0.5 * ((2 * t - 1) + 1)) * np.power(2, -10 * (2 * t - 1)) + 2))

@jit(cache=True)
def ease_back_in(t: np.ndarray) -> np.ndarray:
    return t * t * t - t * np.sin(t * np.pi)

@jit(cache=True)
def ease_back_out(t: np.ndarray) -> np.ndarray:
    p = 1 - t
    return 1 - (p * p * p - p * np.sin(p * np.pi))

@jit(cache=True)
def ease_back_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * (2 * t) * (2 * t) * (2 * t) - (2 * t) * np.sin((2 * t) * np.pi),
                    0.5 * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) - (1 - (2 * t - 1)) * np.sin((1 - (2 * t - 1)) * np.pi) + 0.5)

@jit(cache=True)
def ease_bounce_in(t: np.ndarray) -> np.ndarray:
    return 1 - ease_bounce_out(1 - t)

@jit(cache=True)
def ease_bounce_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 4 / 11, 121 * t * t / 16,
        np.where(t < 8 / 11, (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0,
        np.where(t < 9 / 10, (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0,
                (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0)))

@jit(cache=True)
def ease_bounce_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * ease_bounce_in(t * 2), 0.5 * ease_bounce_out(t * 2 - 1) + 0.5)

def ease_op(op: EnumEase,
            start: float=0, end: float=1, duration: float=1,
            alpha: float=1., clip: Tuple[int, int]=(0, 1)) -> np.ndarray:
    """
    Compute eased values.

    Parameters:
        op (EaseOP): Easing operator.
        start (float): Starting value.
        end (float): Ending value.
        duration (float): Duration of the easing.
        alpha (float): Alpha values.
        clip (Tuple[int, int]): Clip range.

    Returns:
        TYPE_NUMBER: Eased value(s)
    """
    if (func := getattr(MODULE, f"ease_{op.name.lower()}", None)) is None:
        raise BadOperatorException(op.name)
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
    COS = 3
    TAN = 6
    SAWTOOTH = 30
    TRIANGLE = 32
    SQUARE = 34
    PULSE = 36
    RAMP = 40
    STEP = 41
    EXPONENTIAL = 50
    LOGARITHMIC = 55
    NOISE = 60
    HAVERSINE = 70
    RECTANGULAR_PULSE = 80
    GAUSSIAN = 90
    CHIRP = 100

@jit(cache=True)
def wave_sin(phase: float, frequency: float, amplitude: float, offset: float,
             timestep: float) -> float:
    return amplitude * np.sin(frequency * np.pi * 2 * timestep + phase) + offset

@jit(cache=True)
def wave_cos(phase: float, frequency: float, amplitude: float, offset: float,
             timestep: float) -> float:
    return amplitude * np.cos(frequency * np.pi * 2 * timestep + phase) + offset

@jit(cache=True)
def wave_tan(phase: float, frequency: float, amplitude: float, offset: float,
             timestep: float) -> float:
    return amplitude * np.tan(frequency * np.pi * 2 * timestep + phase) + offset

@jit(cache=True)
def wave_sawtooth(phase: float, frequency: float, amplitude: float, offset: float,
                  timestep: float) -> float:
    return amplitude * (2 * (frequency * timestep + phase) % 1 - 0.5) + offset

@jit(cache=True)
def wave_triangle(phase: float, frequency: float, amplitude: float, offset: float,
                  timestep: float) -> float:
    return amplitude * (4 * np.abs((frequency * timestep + phase) % 1 - 0.5) - 1) + offset

@jit(cache=True)
def wave_ramp(phase: float, frequency: float, amplitude: float, offset: float,
              timestep: float) -> float:
    return amplitude * (frequency * timestep + phase % 1) + offset

@jit(cache=True)
def wave_step(phase: float, frequency: float, amplitude: float, offset: float,
              timestep: float) -> float:
    return amplitude * np.heaviside(frequency * timestep + phase, 1) + offset

@jit(cache=True)
def wave_haversine(phase: float, frequency: float, amplitude: float, offset: float,
                   timestep: float) -> float:
    return amplitude * (1 - np.cos(frequency * np.pi * 2 * (timestep + phase))) + offset

@jit(cache=True)
def wave_noise(phase: float, frequency: float, amplitude: float, offset: float,
               timestep: float) -> float:
    return amplitude * np.random.uniform(-1, 1) + offset

# =============================================================================
# === WAVE FUNCTIONS COMPLEX ===
# =============================================================================

@jit(cache=True)
def wave_square(phase: float, frequency: float, amplitude: float, offset: float,
                timestep: float) -> float:
    return amplitude * np.sign(np.sin(np.pi * 2 * timestep + phase) - frequency) + offset

@jit(cache=True)
def wave_exponential(phase: float, frequency: float, amplitude: float,
                     offset: float, timestep: float) -> float:
    return amplitude * np.exp(-frequency * (timestep + phase)) + offset

@jit(cache=True)
def wave_rectangular_pulse(phase: float, frequency: float, amplitude: float,
                           offset: float, timestep: float) -> float:
    return amplitude * np.heaviside(timestep + phase, 1) * np.heaviside(-(timestep + phase) + frequency, 1) + offset

@jit(cache=True)
def wave_logarithmic(phase: float, frequency: float, amplitude: float, offset: float,
                     timestep: float) -> float:
    return amplitude * np.log10(timestep + phase) / np.max(1, np.log10(frequency)) + offset

@jit(cache=True)
def wave_chirp(phase: float, frequency: float, amplitude: float, offset: float,
               timestep: float) -> float:
    return amplitude * np.sin(np.pi * 2 * frequency * (timestep + phase)**2) + offset

####

@jit(cache=True)
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
