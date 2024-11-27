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

MIN_IMAGE_SIZE = 32

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

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
    COORD2D = 22
    STRING = 0
    LIST = 2
    DICT = 3
    IMAGE = 4
    LATENT = 5
    # ENUM = 6
    ANY = 9
    MASK = 7
    # MIXLAB LAYER
    LAYER = 8

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

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

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

def load_file(fname: str) -> str | None:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(e)

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
                val = parse_param(data, k, typ, default)
                vals.append(val)
                found = True
                break

        if found is None:
            fail += 1

    return vals

def decode_tensor(tensor: torch.Tensor) -> str:
    if tensor.ndim > 3:
        b, h, w, cc = tensor.shape
    else:
        cc = 1
        b, h, w = tensor.shape
    return f"{b}x{w}x{h}x{cc}"

def parse_value(val:Any, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0) -> List[Any]:
    """Convert target value into the new specified type."""

    if typ == EnumConvertType.ANY:
        return val

    if isinstance(default, torch.Tensor) and typ not in [EnumConvertType.IMAGE,
                                                         EnumConvertType.MASK,
                                                         EnumConvertType.LATENT]:
        h, w = default.shape[:2]
        cc = default.shape[2] if len(default.shape) > 2 else 1
        default = (w, h, cc)

    if val is None:
        if default is None:
            return None
        val = default

    if isinstance(val, dict):
        # old jovimetrix index?
        if '0' in val or 0 in val:
            val = [val.get(i, val.get(str(i), 0)) for i in range(min(len(val), 4))]
        # coord2d?
        elif 'x' in val:
            val = [val.get(c, 0) for c in 'xyzw']
        # wacky color struct?
        elif 'r' in val:
            val = [val.get(c, 0) for c in 'rgba']
    elif isinstance(val, torch.Tensor) and typ not in [EnumConvertType.IMAGE,
                                                       EnumConvertType.MASK,
                                                       EnumConvertType.LATENT]:
        h, w = val.shape[:2]
        cc = val.shape[2] if len(val.shape) > 2 else 1
        val = (w, h, cc)

    new_val = val
    if typ in [EnumConvertType.FLOAT, EnumConvertType.INT,
            EnumConvertType.VEC2, EnumConvertType.VEC2INT,
            EnumConvertType.VEC3, EnumConvertType.VEC3INT,
            EnumConvertType.VEC4, EnumConvertType.VEC4INT,
            EnumConvertType.COORD2D]:

        if not isinstance(val, (list, tuple, torch.Tensor)):
            val = [val]

        size = max(1, int(typ.value / 10))
        new_val = []
        for idx in range(size):
            try:
                d = default[idx] if idx < len(default) else 0
            except:
                try:
                    d = default.get(str(idx), 0)
                except:
                    d = default

            v = d if val is None else val[idx] if idx < len(val) else d
            if isinstance(v, (str, )):
                v = v.strip('\n').strip()
                if v == '':
                    v = 0

            try:
                if typ in [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4]:
                    v = round(float(v or 0), 16)
                else:
                    v = int(v)
                if clip_min is not None:
                    v = max(v, clip_min)
                if clip_max is not None:
                    v = min(v, clip_max)
            except Exception as e:
                logger.exception(e)
                logger.error(f"Error converting value: {val} -- {v}")
                v = 0

            if v == 0:
                v = zero
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
                if not isinstance(new_val, (list, tuple,)):
                    new_val = [new_val]
                new_val = {i: v for i, v in enumerate(new_val)}
        except Exception as e:
            logger.exception(e)
    elif typ == EnumConvertType.LIST:
        new_val = list(new_val)
    elif typ == EnumConvertType.STRING:
        if isinstance(new_val, (str, list, int, float,)):
            new_val = [new_val]
        new_val = ", ".join(map(str, new_val)) if not isinstance(new_val, str) else new_val
    elif typ == EnumConvertType.BOOLEAN:
        if isinstance(new_val, (torch.Tensor,)):
            new_val = True
        elif isinstance(new_val, (dict,)):
            new_val = len(new_val.keys()) > 0
        elif isinstance(new_val, (list, tuple,)) and len(new_val) > 0 and (nv := new_val[0]) is not None:
            if isinstance(nv, (bool, str,)):
                new_val = bool(nv)
            elif isinstance(nv, (int, float,)):
                new_val = nv > 0
    elif typ == EnumConvertType.LATENT:
        # covert image into latent
        if isinstance(new_val, (torch.Tensor,)):
            new_val = {'samples': new_val.unsqueeze(0)}
        else:
            # convert whatever into a latent sample...
            new_val = torch.empty((4, 64, 64), dtype=torch.uint8).unsqueeze(0)
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
            new_val = torch.empty((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 1), dtype=torch.uint8)
            new_val[0,:,:] = color

    elif issubclass(typ, Enum):
        new_val = typ[val]

    if typ == EnumConvertType.COORD2D:
        new_val = {'x': new_val[0], 'y': new_val[1]}
    return new_val

def parse_param(data:dict, key:str, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0, skip_list=False) -> List[Any]:
    """Convenience because of the dictionary parameters.
    Convert list of values into a list of specified type.
    """
    val = data.get(key, default)
    if typ == EnumConvertType.ANY:
        if isinstance(val, (list,)):
            val = tuple([val])
        elif val is None:
            val = [default]
        #return val

    if isinstance(val, (str,)):
        try: val = json.loads(val.replace("'", '"'))
        except json.JSONDecodeError: pass
    # see if we are a Jovimetrix hacked vector blob... {0:x, 1:y, 2:z, 3:w}
    elif isinstance(val, dict):
        # mixlab layer?
        if (image := val.get('image', None)) is not None:
            ret = image
            if (mask := val.get('mask', None)) is not None:
                while len(mask.shape) < len(image.shape):
                    mask = mask.unsqueeze(-1)
                ret = torch.cat((image, mask), dim=-1)
            if ret.ndim > 3:
                val = [t for t in ret]
            elif ret.ndim == 3:
                val = [v.unsqueeze(-1) for v in ret]
        # vector patch....
        elif 'xyzw' in val:
            val = tuple(x for x in val["xyzw"])
        # latents....
        elif 'samples' in val:
            val = tuple(x for x in val["samples"])
        elif ('0' in val) or (0 in val):
            val = tuple(val.get(i, val.get(str(i), 0)) for i in range(min(len(val), 4)))
        elif 'x' in val and 'y' in val:
            val = tuple(val.get(c, 0) for c in 'xyzw')
        elif 'r' in val and 'g' in val:
            val = tuple(val.get(c, 0) for c in 'rgba')
        elif len(val) == 0:
            val = tuple()
    elif isinstance(val, (torch.Tensor,)):
        # a batch of RGB(A)
        if val.ndim > 3:
            val = [t for t in val]
        # a batch of Grayscale
        else:
            val = [t.unsqueeze(-1) for t in val]
    elif isinstance(val, (list, tuple, set)):
        if len(val) == 0:
            val = [None]
        elif isinstance(val, (tuple, set,)):
            if skip_list == False:
                val = [val]
            else:
                val = val[0][0]
    elif issubclass(type(val), (Enum,)):
        val = [str(val.name)]

    if not isinstance(val, (list,)):
        val = [val]
    return [parse_value(v, typ, default, clip_min, clip_max, zero) for v in val]

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
