"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
UTIL support
"""

import os
import math
from enum import Enum
from typing import Any, List, Generator, Optional, Tuple, Union

import torch

from loguru import logger

# =============================================================================
# === ENUMERATION ===
# =============================================================================

class EnumConvertType(Enum):
    STRING = 0
    BOOLEAN = 1
    INT = 12
    FLOAT = 14
    VEC2 = 20
    VEC2INT = 25
    VEC3 = 30
    VEC3INT = 35
    VEC4 = 40
    VEC4INT = 45
    LIST = 2
    DICT = 3
    IMAGE = 4
    #LATENT = 5
    #MASK = 6

# =============================================================================
# === SUPPORT ===
# =============================================================================

def parse_dynamic(who, data) -> list:
    vals = []
    count = 1
    while (val := data.get(f"{who}_{count}", None)) is not None:
        vals.append(val)
        count += 1
    return vals

def parse_value(val:List[Any], typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0) -> Any:
    """Convert target list of values into the new specified type."""
    if not isinstance(val, (str, set, list,)):
        if isinstance(val, (dict,)):
            if 'samples' in val:
                # latents....
                val = val["samples"]
            else:
                val = list(val.values())
        elif isinstance(val, (tuple, )):
            val = list(val)
        else:
            val = [val]

    if typ != EnumConvertType.IMAGE and isinstance(val, (torch.Tensor,)):
        val = list(val.size())[1:4] + [val[0]]

    pos = len(val)
    size = max(1, int(typ.value / 10))
    for x in range(size - pos):
        idx = pos + x
        last = val[-1] if len(val) else 0
        val.append(default[idx] if default and idx < len(default) else last)

    # val = [v.value if issubclass(type(v), Enum) else v for v in val]
    if typ in [EnumConvertType.FLOAT, EnumConvertType.INT,
                EnumConvertType.VEC2, EnumConvertType.VEC2INT,
                EnumConvertType.VEC3, EnumConvertType.VEC3INT,
                EnumConvertType.VEC4, EnumConvertType.VEC4INT]:

        for idx in range(size):
            if isinstance(val[idx], str):
                parts = val[idx].split('.', 1)
                if len(parts) > 1:
                    val[idx] ='.'.join(parts[:2])
            elif isinstance(val[idx], (list, tuple, set, dict)):
                val[idx] = 0
                #if isinstance(val[idx], (dict,)):
                #    val[idx] = 1
                #else:
            try:
                if typ in [EnumConvertType.FLOAT, EnumConvertType.VEC2,
                            EnumConvertType.VEC3, EnumConvertType.VEC4]:
                    val[idx] = round(float(val[idx]), 12)
                else:
                    val[idx] = int(val[idx])
            except ValueError as _:
                val = [0] * size
                break

            if clip_min is not None:
                val[idx] = max(val[idx], clip_min)
            if clip_max is not None:
                val[idx] = min(val[idx], clip_max)
            if val[idx] == 0:
                val[idx] = zero
        val = val[:size]
    elif typ == EnumConvertType.IMAGE:
        if isinstance(val, (torch.Tensor,)):
            val = val.tolist()
    elif typ == EnumConvertType.STRING:
        if not isinstance(val, (str,)):
            val = [", ".join([str(v) for v in val])]
    elif typ == EnumConvertType.BOOLEAN:
        val = [True if isinstance(v, (torch.Tensor,)) else bool(v) \
               if v is not None and isinstance(v, (bool, int, float, str, list, set, dict,)) else False for v in val]
    elif typ == EnumConvertType.DICT:
        val = [{i: v for i, v in enumerate(val)}]
    if typ == EnumConvertType.LIST:
        val = [val]
    elif len(val) == 1:
        return val[0]
    return val

def parse_parameter(key: str, data: Union[dict, List[dict]], default: Any,
                    typ: EnumConvertType, clip_min: Optional[float]=None,
                    clip_max: Optional[float]=None, zero:int=0) -> tuple[List[Any]]:
    """Convert all inputs, regardless of structure, into a list of parameters."""
    # should be operating on a list of values, all times
    # typ = EnumConvertType[typ]
    if not isinstance(default, (list, tuple,)):
        default = [default]
    unified = data.get(key, default)
    if not isinstance(unified, (list, )):
        unified = [unified]
    if len(unified) == 0:
        unified = [default[0] if len(default) else 0]
    return [parse_value(u, typ, default, clip_min, clip_max, zero) for u in unified]

def update_nested_dict(d, path, value) -> None:
    keys = path.split('.')
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    last_key = keys[-1]

    # Check if the key already exists
    if last_key in current and isinstance(current[last_key], dict):
        current[last_key].update(value)
    else:
        current[last_key] = value

def zip_longest_fill(*iterables: Any) -> Generator[Tuple[Any, ...], None, None]:
    """
    Zip longest with fill value.

    This function behaves like itertools.zip_longest, but it fills the values
    of exhausted iterators with their own last values instead of None.
    """
    try:
        iterators = [iter(iterable) for iterable in iterables]
    except Exception as e:
        logger.error(iterables)
        logger.error(str(e))

    while True:
        values = [next(iterator, None) for iterator in iterators]

        # Check if all iterators are exhausted
        if all(value is None for value in values):
            break

        # Fill in the last values of exhausted iterators with their own last values
        for i, _ in enumerate(iterators):
            if values[i] is None:
                iterator_copy = iter(iterables[i])
                while True:
                    current_value = next(iterator_copy, None)
                    if current_value is None:
                        break
                    values[i] = current_value

        yield tuple(values)

def deep_merge_dict(*dicts: dict) -> dict:
    """
    Deep merge multiple dictionaries recursively.

    Args:
        *dicts: Variable number of dictionaries to be merged.

    Returns:
        dict: Merged dictionary.
    """
    def _deep_merge(d1: Any, d2: Any) -> Any:
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            return d2

        merged_dict = d1.copy()

        for key in d2:
            if key in merged_dict:
                if isinstance(merged_dict[key], dict) and isinstance(d2[key], dict):
                    merged_dict[key] = _deep_merge(merged_dict[key], d2[key])
                elif isinstance(merged_dict[key], list) and isinstance(d2[key], list):
                    merged_dict[key].extend(d2[key])
                else:
                    merged_dict[key] = d2[key]
            else:
                merged_dict[key] = d2[key]
        return merged_dict

    merged = {}
    for d in dicts:
        merged = _deep_merge(merged, d)
    return merged

def grid_make(data: List[Any]) -> Tuple[List[List[Any]], int, int]:
    """
    Create a 2D grid from a 1D list.

    Args:
        data (List[Any]): Input data.

    Returns:
        Tuple[List[List[Any]], int, int]: A tuple containing the 2D grid, number of columns,
        and number of rows.
    """
    size = len(data)
    grid = int(math.sqrt(size))
    if grid * grid < size:
        grid += 1
    if grid < 1:
        return [], 0, 0

    rows = size // grid
    if size % grid != 0:
        rows += 1

    ret = []
    cols = 0
    for j in range(rows):
        end = min((j + 1) * grid, len(data))
        cols = max(cols, end - j * grid)
        d = [data[i] for i in range(j * grid, end)]
        ret.append(d)
    return ret, cols, rows

def path_next(pattern: str) -> str:
    """
    Finds the next free path in an sequentially named list of files
    """
    i = 1
    while os.path.exists(pattern % i):
        i = i * 2

    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2
        a, b = (c, b) if os.path.exists(pattern % c) else (a, c)
    return pattern % b
