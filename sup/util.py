"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
UTIL support
"""

import os
import json
import math
from enum import Enum
from typing import Any, List, Generator, Optional, Tuple

import numpy as np
import torch

from loguru import logger

MIN_IMAGE_SIZE = 32

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
    # ENUM = 6
    COORD2D = 22
    ANY = 9
    MASK = 7

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

def parse_dynamic(data:dict, prefix:str, typ:EnumConvertType, default: Any) -> List[Any]:
    """Convert iterated input field(s) based on a s into a single compound list of entries.

    The default will just look for all keys as integer:

        `#_<field name>` or `#_<prefix>_<field name>`

    This will return N entries in a list based on the prefix pattern or not.

    """
    vals = []
    fail = 0
    keys = data.keys()
    for i in range(100):
        if fail > 2:
            break

        found = None
        for k in keys:
            if k.startswith(f"{i}_") or k.startswith(f"{i}_{prefix}_"):
                found = k
                break

        if found is None:
            fail += 1
            continue

        val = parse_param(data, found, typ, default)
        vals.append(val)
    return vals

def parse_value(val:Any, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0) -> List[Any]:
    """Convert target value into the new specified type."""

    if isinstance(default, torch.Tensor) and typ not in [EnumConvertType.ANY, EnumConvertType.IMAGE, EnumConvertType.LATENT]:
        h, w = default.shape[:2]
        cc = default.shape[2] if len(default.shape) > 2 else 1
        default = (w, h, cc)

    if val is None:
        if default is None:
            return None
        val = default

    if isinstance(val, dict):
        if ('0' in val and '1' in val) or (0 in val and 1 in val):
            val = [val.get(i, val.get(str(i), 0)) for i in range(min(len(val), 4))]
        elif 'x' in val and 'y' in val:
            val = [val.get(c, 0) for c in 'xyzw']
        elif 'r' in val and 'g' in val:
            val = [val.get(c, 0) for c in 'rgba']
    elif isinstance(val, torch.Tensor) and typ not in [EnumConvertType.ANY, EnumConvertType.IMAGE, EnumConvertType.LATENT]:
        h, w = val.shape[:2]
        cc = val.shape[2] if len(val.shape) > 2 else 1
        val = (w, h, cc)

    if val is not None: #and typ not in [EnumConvertType.ANY]:
        if not isinstance(val, (list, tuple, torch.Tensor)):
            val = [val]

    new_val = val
    if typ in [EnumConvertType.FLOAT, EnumConvertType.INT,
            EnumConvertType.VEC2, EnumConvertType.VEC2INT,
            EnumConvertType.VEC3, EnumConvertType.VEC3INT,
            EnumConvertType.VEC4, EnumConvertType.VEC4INT,
            EnumConvertType.COORD2D]:

        size = max(1, int(typ.value / 10))
        new_val = []
        for idx in range(size):
            d = default[idx] if isinstance(default, (list, tuple, set, dict, torch.Tensor)) and idx < len(default) else default
            v = d if val is None else val[idx] if idx < len(val) else d
            try:
                if typ in [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4]:
                    v = round(float(v), 16)
                else:
                    v = int(v)
                if clip_min is not None:
                    v = max(v, clip_min)
                if clip_max is not None:
                    v = min(v, clip_max)
                if v == 0:
                    v = zero
            except Exception as e:
                logger.exception(e)
                logger.error(f"Error converting value: {e}")
                v = 0
            new_val.append(v)
        new_val = new_val[0] if size == 1 else tuple(new_val)
    elif typ == EnumConvertType.DICT:
        try:
            if isinstance(new_val, (str,)):
                try:
                    new_val = json.loads(new_val)
                except json.decoder.JSONDecodeError:
                    new_val = {}
            else:
                new_val = {i: v for i, v in enumerate(new_val)}
        except Exception as e:
            logger.exception(e)
    elif typ == EnumConvertType.LIST:
        new_val = list(new_val)
    elif typ == EnumConvertType.STRING:
        new_val = ", ".join(map(str, new_val)) if not isinstance(new_val, str) else new_val
    elif typ == EnumConvertType.BOOLEAN:
        ret = False
        if isinstance(new_val, (torch.Tensor,)):
            ret = True
        elif isinstance(new_val, (dict,)):
            ret = len(new_val.keys()) > 0
        elif isinstance(new_val, (list, tuple,)) and len(new_val) > 0 and (nv := new_val[0]) is not None:
            if isinstance(nv, (bool, str,)):
                ret = bool(nv)
            elif isinstance(nv, (int, float,)):
                ret = nv > 0
        new_val = ret
    elif typ == EnumConvertType.LATENT:
        # covert image into latent
        if isinstance(new_val, (torch.Tensor,)):
            new_val = {'samples': new_val.unsqueeze(0)}
        else:
            # convert whatever into a latent sample...
            new_val = torch.empty((4, 512, 512), dtype=torch.uint8).unsqueeze(0)
            new_val = {'samples': new_val}
    elif typ == EnumConvertType.IMAGE:
        # covert image into image? just skip if already an image
        if not isinstance(new_val, (torch.Tensor,)):
            color = parse_value(new_val, EnumConvertType.VEC4INT, (0,0,0,255), 0, 255)
            color = torch.tensor(color, dtype=torch.int32).tolist()
            new_val = torch.empty((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 4), dtype=torch.uint8)
            new_val[0,:,:] = color[0]
            new_val[1,:,:] = color[1]
            new_val[2,:,:] = color[2]
            new_val[3,:,:] = color[3]
    elif typ == EnumConvertType.MASK:
        # @TODO: FIX FOR MULTI-CHAN?
        if not isinstance(new_val, (torch.Tensor,)):
            color = parse_value(new_val, EnumConvertType.INT, 0, 0, 255)
            color = torch.tensor(color, dtype=torch.int32).tolist()
            new_val = torch.empty((512, 512, 1), dtype=torch.uint8)
            new_val[0,:,:] = color
        else:
            cc, h, w = new_val.shape
            if cc > 1:
                weights = [0.2989, 0.5870, 0.1140]
                new_val = np.dot(new_val[..., :3], weights)
                new_val = new_val.reshape(512, 512, 1)

    if typ == EnumConvertType.COORD2D:
        new_val = {'x': new_val[0], 'y': new_val[1]}
    return new_val

def parse_param(data:dict, key:str, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0) -> List[Any]:
    """Convenience because of the dictionary parameters.
    Convert list of values into a list of specified type.
    """
    val = data.get(key, default)
    if isinstance(val, (str,)):
        try: val = json.loads(val.replace("'", '"'))
        except json.JSONDecodeError: pass
    # see if we are a Jovimetrix hacked vector blob... {0:x, 1:y, 2:z, 3:w}
    elif isinstance(val, dict):
        # latents....
        if 'samples' in val:
            val = tuple(x for x in val["samples"])
        elif ('0' in val) or (0 in val):
            val = tuple(val.get(i, val.get(str(i), 0)) for i in range(min(len(val), 4)))
        elif 'x' in val and 'y' in val:
            val = tuple(val.get(c, 0) for c in 'xyzw')
        elif 'r' in val and 'g' in val:
            val = tuple(val.get(c, 0) for c in 'rgba')
        elif len(val) == 0:
            logger.debug(f"[parse_param] {val}")
            val = tuple()
    elif isinstance(val, (torch.Tensor,)):
        if val.ndim > 3:
            val = [t for t in val]
        else:
            while (val.ndim < 3):
                val = val.unsqueeze(-1)
    elif isinstance(val, (list, tuple, set)):
        if len(val) == 0:
            val = [None]
    elif issubclass(type(val), (Enum,)):
        val = [str(val.name)]
    if typ == EnumConvertType.ANY:
        return [val]
    if not isinstance(val, (list,)):
        val = [val]
    return [parse_value(v, typ, default, clip_min, clip_max, zero) for v in val]

def vector_swap(pA: Any, pB: Any, swap_x: EnumSwizzle, x:float, swap_y:EnumSwizzle, y:float,
                swap_z:EnumSwizzle, z:float, swap_w:EnumSwizzle, w:float) -> List[float]:
    """Swap out a vector's values with another vector's values, or a constant fill."""
    def parse(target, targetB, swap, val) -> float:
        if swap == EnumSwizzle.CONSTANT:
            return val
        if swap in [EnumSwizzle.B_X, EnumSwizzle.B_Y, EnumSwizzle.B_Z, EnumSwizzle.B_W]:
            target = targetB
        swap = int(swap.value / 10)
        return target[swap] if swap < len(target) else 0

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
