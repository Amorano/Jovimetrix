"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Animation Support
"""

import math
from enum import Enum
from typing import Optional

import numpy as np

__all__ = ["Ease", "Wave"]

HALFPI = math.pi / 2
TAU = math.pi * 2

# =============================================================================
# === EXCEPTIONAL ===
# =============================================================================

class BadOperatorException(Exception):
    """Exception for bad operators."""
    pass

# =============================================================================
# === EASING ===
# =============================================================================

class EnumEase(Enum):
    QUAD_IN = 0
    QUAD_OUT = 1
    QUAD_IN_OUT = 2

    CUBIC_IN = 3
    CUBIC_OUT = 4
    CUBIC_IN_OUT = 5

    QUARTIC_IN = 6
    QUARTIC_OUT = 7
    QUARTIC_IN_OUT = 8

    QUINTIC_IN = 9
    QUINTIC_OUT = 10
    QUINTIC_IN_OUT = 11

    SIN_IN = 12
    SIN_OUT = 13
    SIN_IN_OUT = 14

    CIRCULAR_IN = 15
    CIRCULAR_OUT = 16
    CIRCULAR_IN_OUT = 17

    EXPONENTIAL_IN = 18
    EXPONENTIAL_OUT = 19
    EXPONENTIAL_IN_OUT = 20

    ELASTIC_IN = 21
    ELASTIC_OUT = 22
    ELASTIC_IN_OUT = 23

    BACK_IN = 24
    BACK_OUT = 25
    BACK_IN_OUT = 26

    BOUNCE_IN = 27
    BOUNCE_OUT = 28
    BOUNCE_IN_OUT = 29

class Ease:
    @classmethod
    def quad_in(cls, t: np.ndarray) -> np.ndarray:
        return t * t

    @classmethod
    def quad_out(cls, t: np.ndarray) -> np.ndarray:
        return -(t * (t - 2))

    @classmethod
    def quad_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 2 * t * t, (-2 * t * t) + (4 * t) - 1)

    @classmethod
    def cubic_in(cls, t: np.ndarray) -> np.ndarray:
        return t * t * t

    @classmethod
    def cubic_out(cls, t: np.ndarray) -> np.ndarray:
        return (t - 1) * (t - 1) * (t - 1) + 1

    @classmethod
    def cubic_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 4 * t * t * t,
                        0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

    @classmethod
    def quartic_in(cls, t: np.ndarray) -> np.ndarray:
        return t * t * t * t

    @classmethod
    def quartic_out(cls, t: np.ndarray) -> np.ndarray:
        return (t - 1) * (t - 1) * (t - 1) * (1 - t) + 1

    @classmethod
    def quartic_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 8 * t * t * t * t,
                        -8 * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1)

    @classmethod
    def quintic_in(cls, t: np.ndarray) -> np.ndarray:
        return t * t * t * t * t

    @classmethod
    def quintic_out(cls, t: np.ndarray) -> np.ndarray:
        return (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1

    @classmethod
    def quintic_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 16 * t * t * t * t * t,
                        0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

    @classmethod
    def sin_in(cls, t: np.ndarray) -> np.ndarray:
        return np.sin((t - 1) * HALFPI) + 1

    @classmethod
    def sin_out(cls, t: np.ndarray) -> np.ndarray:
        return np.sin(t * HALFPI)

    @classmethod
    def sin_in_out(cls, t: np.ndarray) -> np.ndarray:
        return 0.5 * (1 - np.cos(t * math.pi))

    @classmethod
    def circular_in(cls, t: np.ndarray) -> np.ndarray:
        return 1 - np.sqrt(1 - (t * t))

    @classmethod
    def circular_out(cls, t: np.ndarray) -> np.ndarray:
        return np.sqrt((2 - t) * t)

    @classmethod
    def circular_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 0.5 * (1 - np.sqrt(1 - 4 * (t * t))),
                        0.5 * (np.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1))

    @classmethod
    def exponential_in(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t == 0, 0, np.power(2, 10 * (t - 1)))

    @classmethod
    def exponential_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t == 1, 1, 1 - np.power(2, -10 * t))

    @classmethod
    def exponential_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t == 0, t, np.where(t < 0.5, 0.5 * np.power(2, (20 * t) - 10),
                                            -0.5 * np.power(2, (-20 * t) + 10) + 1))

    @classmethod
    def elastic_in(cls, t: np.ndarray) -> np.ndarray:
        return np.sin(13 * HALFPI * t) * np.power(2, 10 * (t - 1))

    @classmethod
    def elastic_out(cls, t: np.ndarray) -> np.ndarray:
        return np.sin(-13 * HALFPI * (t + 1)) * np.power(2, -10 * t) + 1

    @classmethod
    def elastic_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 0.5 * np.sin(13 * HALFPI * (2 * t)) * np.power(2, 10 * ((2 * t) - 1)),
                        0.5 * (np.sin(-13 * HALFPI * ((2 * t - 1) + 1)) * np.power(2, -10 * (2 * t - 1)) + 2))

    @classmethod
    def back_in(cls, t: np.ndarray) -> np.ndarray:
        return t * t * t - t * np.sin(t * math.pi)

    @classmethod
    def back_out(cls, t: np.ndarray) -> np.ndarray:
        p = 1 - t
        return 1 - (p * p * p - p * np.sin(p * math.pi))

    @classmethod
    def back_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 0.5 * (2 * t) * (2 * t) * (2 * t) - (2 * t) * np.sin((2 * t) * math.pi),
                        0.5 * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) - (1 - (2 * t - 1)) * np.sin((1 - (2 * t - 1)) * math.pi) + 0.5)

    @classmethod
    def bounce_in(cls, t: np.ndarray) -> np.ndarray:
        return 1 - cls.bounce_out(1 - t)

    @classmethod
    def bounce_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 4 / 11, 121 * t * t / 16,
            np.where(t < 8 / 11, (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0,
            np.where(t < 9 / 10, (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0,
                    (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0)))

    @classmethod
    def bounce_in_out(cls, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 0.5 * cls.bounce_in(t * 2), 0.5 * cls.bounce_out(t * 2 - 1) + 0.5)

    def ease(op: EnumEase,
             start: float=0, end: float=1, duration: float=1,
             alpha: Optional[np.ndarray[np.uint8]]=None,
             clip: tuple[int, int]=(0, 1)) -> np.ndarray:
        """
        Compute eased values.

        Parameters:
            op (EaseOP): Easing operator.
            start (float): Starting value.
            end (float): Ending value.
            duration (float): Duration of the easing.
            alpha (np.ndarray): Alpha values.
            clip (tuple[int, int]): Clip range.

        Returns:
            np.ndarray: Eased values.
        """

        if (func := getattr(Ease, op.name.lower(), None)) is None:
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

class EnumWaveSimple(Enum):
    SIN = EnumWave.SIN
    SIN_INV = EnumWave.SIN_INV
    SIN_ABS = EnumWave.SIN_ABS
    COS = EnumWave.COS
    COS_INV = EnumWave.COS_INV
    COS_ABS = EnumWave.COS_ABS
    SAWTOOTH = EnumWave.SAWTOOTH
    TRIANGLE = EnumWave.TRIANGLE
    RAMP = EnumWave.RAMP
    STEP = EnumWave.STEP
    HAVERSINE = EnumWave.HAVERSINE
    NOISE = EnumWave.NOISE

    """
        "SQUARE": comp.wave_square,
        "PULSE": comp.wave_pulse,
        "EXP": comp.wave_exponential,
        "RECT PULSE": comp.wave_rectangular_pulse,

        "LOG": comp.wave_logarithmic,
        "GAUSSIAN": comp.wave_gaussian,
        "CHIRP": comp.wave_chirp_signal,
    }
    """

class Wave:
    @classmethod
    def sin(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return amplitude * np.sin(TAU * timestep + phase) + offset

    @classmethod
    def sin_inv(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return -amplitude * np.sin(TAU * timestep + phase) + offset

    @classmethod
    def sin_abs(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return np.abs(amplitude * np.sin(TAU * timestep + phase)) + offset

    @classmethod
    def cos(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return amplitude * np.cos(TAU * timestep + phase) + offset

    @classmethod
    def cos_inv(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return -amplitude * np.cos(TAU * timestep + phase) + offset

    @classmethod
    def cos_abs(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return np.abs(amplitude * np.cos(TAU * timestep + phase)) + offset

    @classmethod
    def sawtooth(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return amplitude * (2 * (timestep + phase) % 1 - 0.5) + offset

    @classmethod
    def triangle(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return amplitude * (4 * np.abs((timestep + phase) % 1 - 0.5) - 1) + offset

    @classmethod
    def ramp(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return amplitude * (timestep + phase % 1) + offset

    @classmethod
    def step(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return amplitude * np.heaviside(timestep + phase, 1) + offset

    @classmethod
    def haversine(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return amplitude * (1 - np.cos(TAU * (timestep + phase))) + offset

    @classmethod
    def noise(cls, phase: float, amplitude: float, offset: float, timestep: float) -> float:
        return amplitude * np.random.uniform(-1, 1) + offset

    # =============================================================================
    # === WAVE FUNCTIONS COMPLEX ===
    # =============================================================================

    @classmethod
    def square(cls, phase: float, amplitude: float, offset: float, timestep: float, duty_cycle: float = 0.5) -> float:
        return amplitude * np.sign(np.sin(TAU * timestep + phase) - duty_cycle) + offset

    @classmethod
    def pulse(cls, phase: float, amplitude: float, offset: float, timestep: float, duty_cycle: float = 0.5) -> float:
        return amplitude * np.sign(np.sin(TAU * timestep + phase) - duty_cycle) + offset

    @classmethod
    def exponential(cls, phase: float, amplitude: float, offset: float, timestep: float, decay: float = 1.0) -> float:
        return amplitude * np.exp(-decay * (timestep + phase)) + offset

    @classmethod
    def rectangular_pulse(cls, phase: float, amplitude: float, offset: float, timestep: float, pulse_width: float = 0.1) -> float:
        return amplitude * np.heaviside(timestep + phase, 1) * np.heaviside(-(timestep + phase) + pulse_width, 1) + offset

    ####

    @classmethod
    def logarithmic(cls, phase: float, amplitude: float, offset: float, timestep: float, base: float = 10) -> float:
        return amplitude * np.log10(timestep + phase) / np.log10(base) + offset

    @classmethod
    def gaussian(cls, phase: float, amplitude: float, offset: float, timestep: float, mean: float = 0, std_dev: float = 1) -> float:
        return amplitude * np.exp(-0.5 * ((timestep + phase - mean) / std_dev)**2) + offset

    @classmethod
    def chirp(cls, phase: float, amplitude: float, offset: float, timestep: float, frequency_slope: float = 1.0) -> float:
        return amplitude * np.sin(TAU * frequency_slope * (timestep + phase)**2) + offset

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    alpha_values = np.linspace(0, 1, 5)
    for op in EnumEase:
        # Logger.debug(op)
        result = Ease.ease(op, start=0, end=1, duration=1, alpha=alpha_values, clip=(0, 1))
        print(f"{op}: {result}")
