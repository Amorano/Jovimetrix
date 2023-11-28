"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix

                    Copyright 2023 Alexander Morano (Joviex)

Animation Supports.
"""

import math
from enum import Enum

import numpy as np

__all__ = ["ease", "EaseOP"]

HALFPI = math.pi / 2
TAU = math.pi * 2

# command OPs for EASE functions
_OP_EASE = {}

# =============================================================================
# === EASING ===
# =============================================================================

class BadOperatorException(Exception):
    """Exception for bad operators."""
    pass

class EaseOP(Enum):
    """Enum class for easing operators."""
    pass

def ease(op: EaseOP, start: float=0, end: float=1, duration: float=1, alpha: np.ndarray=None, clip: tuple[int, int]=(0, 1)) -> np.ndarray:
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
    if (func := _OP_EASE.get(op.name, None)) is None:
        raise BadOperatorException(str(op))

    t = clip[0] * (1 - alpha) + clip[1] * alpha
    duration = max(min(duration, 1), 0)
    t /= duration
    a = func(t)
    return end * a + start * (1 - a)

# =============================================================================
# === LINEAR ===
# =============================================================================

def quad_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 2 * t * t, (-2 * t * t) + (4 * t) - 1)

def quad_in(t: np.ndarray) -> np.ndarray:
    return t * t

def quad_out(t: np.ndarray) -> np.ndarray:
    return -(t * (t - 2))

def cubic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t

def cubic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) + 1

def cubic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 4 * t * t * t,
                    0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

def quartic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t * t

def quartic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) * (1 - t) + 1

def quartic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 8 * t * t * t * t,
                    -8 * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1)

def quintic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t * t * t

def quintic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1

def quintic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 16 * t * t * t * t * t,
                    0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

def sine_in(t: np.ndarray) -> np.ndarray:
    return np.sin((t - 1) * HALFPI) + 1

def sine_out(t: np.ndarray) -> np.ndarray:
    return np.sin(t * HALFPI)

def sine_in_out(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1 - np.cos(t * math.pi))

def circular_in(t: np.ndarray) -> np.ndarray:
    return 1 - np.sqrt(1 - (t * t))

def circular_out(t: np.ndarray) -> np.ndarray:
    return np.sqrt((2 - t) * t)

def circular_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * (1 - np.sqrt(1 - 4 * (t * t))),
                    0.5 * (np.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1))

def exponential_in(t: np.ndarray) -> np.ndarray:
    return np.where(t == 0, 0, np.power(2, 10 * (t - 1)))

def exponential_out(t: np.ndarray) -> np.ndarray:
    return np.where(t == 1, 1, 1 - np.power(2, -10 * t))

def exponential_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t == 0, t, np.where(t < 0.5, 0.5 * np.power(2, (20 * t) - 10),
                                        -0.5 * np.power(2, (-20 * t) + 10) + 1))

def elastic_in(t: np.ndarray) -> np.ndarray:
    return np.sin(13 * HALFPI * t) * np.power(2, 10 * (t - 1))

def elastic_out(t: np.ndarray) -> np.ndarray:
    return np.sin(-13 * HALFPI * (t + 1)) * np.power(2, -10 * t) + 1

def elastic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * np.sin(13 * HALFPI * (2 * t)) * np.power(2, 10 * ((2 * t) - 1)),
                    0.5 * (np.sin(-13 * HALFPI * ((2 * t - 1) + 1)) * np.power(2, -10 * (2 * t - 1)) + 2))

def back_in(t: np.ndarray) -> np.ndarray:
    return t * t * t - t * np.sin(t * math.pi)

def back_out(t: np.ndarray) -> np.ndarray:
    p = 1 - t
    return 1 - (p * p * p - p * np.sin(p * math.pi))

def back_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * (2 * t) * (2 * t) * (2 * t) - (2 * t) * np.sin((2 * t) * math.pi),
                    0.5 * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) - (1 - (2 * t - 1)) * np.sin((1 - (2 * t - 1)) * math.pi) + 0.5)

def bounce_in(t: np.ndarray) -> np.ndarray:
    return 1 - bounce_out(1 - t)

def bounce_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 4 / 11, 121 * t * t / 16,
                    np.where(t < 8 / 11, (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0,
                             np.where(t < 9 / 10, (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0,
                                      (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0)))

def bounce_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * bounce_in(t * 2), 0.5 * bounce_out(t * 2 - 1) + 0.5)

# =============================================================================
# === REGISTER ENUMS ===
# =============================================================================

def importFunctions() -> None:
    """
    Import easing functions and update EaseOP enum.
    """
    import inspect

    current_frame = inspect.currentframe()
    calling_frame = inspect.getouterframes(current_frame)[0]
    module = inspect.getmodule(calling_frame.frame)
    functs = inspect.getmembers(module, inspect.isfunction)

    idx = 0
    data = {}
    for name, obj in functs:
        if '_in' in name or '_out' in name:
            name = ''.join([e.title() for e in name.split('_')])
            _OP_EASE[name] = obj
            data[name] = idx
        idx += 1

    global EaseOP
    EaseOP  = Enum('DynamicEnum', data)

importFunctions()

if __name__ == "__main__":
    alpha_values = np.linspace(0, 1, 5)
    for op in EaseOP:
        result = ease(op, start=0, end=1, duration=1, alpha=alpha_values, clip=(0, 1))
        print(f"{op.name}: {result}")
