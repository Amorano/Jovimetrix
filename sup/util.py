"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
UTIL support
"""

import os
import json
import math
from enum import Enum
from typing import Any, List, Generator, Optional, Tuple

import torch

from loguru import logger

# =============================================================================
# === ENUMERATION ===
# =============================================================================

class EnumConvertType(Enum):
    BOOLEAN = 1
    FLOAT = 10
    INT = 12
    VEC2 = 20
    VEC2INT = 25
    VEC3 = 30
    VEC3INT = 35
    VEC4 = 40
    VEC4INT = 45
    STRING = 0
    LIST = 2
    DICT = 3
    IMAGE = 4
    LATENT = 5
    ANY = 9
    # ENUM = 6

class EnumSwizzle(Enum):
    A_X = 0
    A_Y = 10
    A_Z = 20
    A_W = 30
    B_X = 9
    B_Y = 11
    B_Z = 21
    B_W = 31
    CONSTANT = 40

# =============================================================================
# === SUPPORT ===
# =============================================================================

def parse_dynamic(who, data) -> list:
    vals = []
    count = 1
    while (val := data.get(f"{who}_{count}", None)) is not None:
        if not isinstance(val, (list, )):
            if isinstance(val, (torch.Tensor,)):
                val = val.tolist()
            else:
                val = [val]
        vals.append(val)
        count += 1
    return vals

def parse_as_list(val: Any) -> List[Any]:
    """Convert value into a list of value."""
    if isinstance(val, (list, tuple,)):
        return val
    if isinstance(val, (str, float, int,)):
        return [val]
    if isinstance(val, (dict,)):
        # latents....
        if 'samples' in val:
            return [v for v in val["samples"]]
        return tuple(list(val.values()))
    if isinstance(val, (torch.Tensor,)):
        if len(val.shape) > 3:
            return [t for t in val]
    if issubclass(type(val), (Enum,)):
        return [[val.name]]
    return [val]

def parse_param(data:dict, key:str, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0, enumType:Any=None) -> List[Any]:
    """Convenience because of the dictionary parameters."""
    val = data.get(key, default)
    return parse_list_value(val, typ, default, clip_min, clip_max, zero, enumType)

def parse_list_value(val:Any|None, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0, enumType:Any=None) -> List[Any]:
    """Convert list of values into a list of specified type."""
    val = default if val is None else val

    # could be a json encoded blob
    if isinstance(val, (str,)):
        try:
            val = json.loads(val.replace("'", '"'))
        except:
            pass

    # see if we are a Jovimetrix hacked vector blob... {0:x, 1:y, 2:z, 3:w}
    if isinstance(val, (dict,)):
        if (x:=val.get('0', None)) is not None and (y:=val.get('1', None)) is not None:
            ret = [x, y]
            if (x:=val.get('2', None)) is not None:
                ret.append(x)
            if (x:=val.get('3', None)) is not None:
                ret.append(x)
            val = (ret,)
        # could be a kijai coord blob
        elif (x:=val.get('x', None)) is not None and (y:=val.get('y', None)) is not None:
            ret = [x, y]
            if (x:=val.get('z', None)) is not None:
                ret.append(x)
            if (x:=val.get('w', None)) is not None:
                ret.append(x)
            val = (ret,)
    elif isinstance(val, (list, tuple,)):
        val = [parse_as_list(v) for v in val]
    else:
        val = parse_as_list(val)
    return [parse_value(v, typ, default, clip_min, clip_max, zero, enumType) for v in val]

def parse_value(val:Any, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0, enumType:Any=None) -> List[Any]:
    """Convert target value into the new specified type."""
    if val is None:
        if default is not None:
            return parse_value(default, typ, default, clip_min, clip_max, zero, enumType)
        return None

    if typ not in [EnumConvertType.ANY, EnumConvertType.IMAGE] and isinstance(val, (torch.Tensor,)):
        val = list(val.size())[1:4] + [val[0]]

    size = 1
    new_val = val
    if typ in [EnumConvertType.FLOAT, EnumConvertType.INT,
            EnumConvertType.VEC2, EnumConvertType.VEC2INT,
            EnumConvertType.VEC3, EnumConvertType.VEC3INT,
            EnumConvertType.VEC4, EnumConvertType.VEC4INT]:

        new_val = []
        if not isinstance(val, (list, tuple,)):
            val = [val]
        last = val[0]
        size = max(1, int(typ.value / 10))
        for idx in range(size):
            v = val[idx] if idx < len(val) else None
            d = new_val[-1] if len(new_val) else val[-1]
            if default is not None:
                d = default
                if isinstance(default, (list, set, tuple,)):
                    if idx < len(default):
                        d = default[idx]
                    else:
                        d = default[-1]
            last = v if v is not None else d
            new_val.append(last)
        """
        if typ in [EnumConvertType.FLOAT, EnumConvertType.INT,
                    EnumConvertType.VEC2, EnumConvertType.VEC2INT,
                    EnumConvertType.VEC3, EnumConvertType.VEC3INT,
                    EnumConvertType.VEC4, EnumConvertType.VEC4INT]:
        """

        for idx in range(size):
            if isinstance(new_val[idx], str):
                parts = new_val[idx].split('.', 1)
                if len(parts) > 1:
                    new_val[idx] ='.'.join(parts[:2])
            elif isinstance(new_val[idx], (list, tuple, set, dict)):
                new_val[idx] = 0
            try:
                if typ in [EnumConvertType.FLOAT, EnumConvertType.VEC2,
                            EnumConvertType.VEC3, EnumConvertType.VEC4]:
                    new_val[idx] = round(float(new_val[idx]), 12)
                else:
                    new_val[idx] = int(new_val[idx])

                if clip_min is not None:
                    new_val[idx] = max(new_val[idx], clip_min)
                if clip_max is not None:
                    new_val[idx] = min(new_val[idx], clip_max)
                if new_val[idx] == 0:
                    new_val[idx] = zero
            except Exception as _:
                try:
                    new_val[idx] = ord(v)
                except:
                    logger.debug(f"value not converted well {val} ... {new_val[idx]} == 0")
                    new_val[idx] = 0
        if size == 1:
            new_val = new_val[0]
        else:
            new_val = new_val[:size]
            new_val = tuple(new_val)
    elif typ == EnumConvertType.IMAGE:
        if isinstance(new_val, (torch.Tensor,)):
            if len(new_val.shape) > 3:
                new_val = [t for t in new_val]
        else:
            # convert whatever into an tensor...
            new_val = torch.empty((4, 512, 512), dtype=torch.uint8)
    elif typ == EnumConvertType.STRING:
        if not isinstance(new_val, (str,)):
            new_val = ", ".join([str(v) for v in new_val])
    elif typ == EnumConvertType.BOOLEAN:
        new_val = True if isinstance(new_val, (torch.Tensor,)) else bool(new_val) \
            if new_val is not None and isinstance(new_val, (bool, int, float, str,)) else False
    elif typ == EnumConvertType.DICT:
        new_val = {i: v for i, v in enumerate(new_val)}
    elif typ == EnumConvertType.LIST:
        new_val = [new_val]
    #elif typ == EnumConvertType.ENUM:
        #new_val = enumType[new_val]
    return new_val

def vector_swap(pA: Any, pB: Any, swap_x: EnumSwizzle, x:float, swap_y:EnumSwizzle, y:float,
                swap_z:EnumSwizzle, z:float, swap_w:EnumSwizzle, w:float) -> List[float]:
    """Swap out a vector's values with another vector's values, or a constant fill."""
    def parse(target, targetB, swap, val) -> float:
        if swap == EnumSwizzle.CONSTANT:
            return val
        if swap in [EnumSwizzle.B_X, EnumSwizzle.B_Y, EnumSwizzle.B_Z, EnumSwizzle.B_W]:
            target = targetB
        swap = int(swap.value / 10)
        return target[swap]

    return [
        parse(pA, pB, swap_x, x),
        parse(pA, pB, swap_y, y),
        parse(pA, pB, swap_z, z),
        parse(pA, pB, swap_w, w)
    ]

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
    else:
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
