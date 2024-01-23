"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
UTIL support
"""

import math
from enum import Enum
from typing import Any, List, Generator, Optional, Tuple, Union

from loguru import logger

# =============================================================================
# === ENUMERATION ===
# =============================================================================

class EnumTupleType(Enum):
    INT = 0
    FLOAT = 1
    STRING = 2
    LIST = 3
    DICT = 4

# =============================================================================
# === SUPPORT ===
# =============================================================================

def convert_parameter(data: Any) -> Any:
    if data is None:
        return [int], [0]

    if not isinstance(data, (list, tuple, set,)):
        data = [data]

    typ = []
    val = []
    for v in data:
        # logger.debug("{} {} {} {}", v, type(v), type(v) == float, type(v) == int)
        t = type(v)
        if t == int:
            t = float
        try:
            v = float(v)
        except Exception as e:
            logger.error(str(e))
            v = 0
        typ.append(t)
        val.append(v)

    return typ, val

def parse_number(key: str, data: Union[dict, List[dict]], typ: EnumTupleType=EnumTupleType.INT, default: tuple[Any]=None, clip_min: Optional[int]=None, clip_max: Optional[int]=None) -> tuple[List[Any]]:
    ret = []
    unified = data.get(key, {})

    if not isinstance(unified, (set, tuple, list,)):
        unified = [unified]

    for v in unified:
        match typ:
            case EnumTupleType.FLOAT:
                if isinstance(v, str):
                    parts = v.split('.', 1)
                    if len(parts) > 1:
                        v ='.'.join(parts[:2])
                v = float(v if v is not None else 0)

            case EnumTupleType.INT:
                v = int(v if v is not None else 0)

        if typ in [EnumTupleType.INT, EnumTupleType.FLOAT]:
            if clip_min is not None:
                v = max(v, clip_min)
            if clip_max is not None:
                v = min(v, clip_max)

        ret.append(v)
    return ret

def parse_tuple(key: str, data: Union[dict, List[dict]], typ: EnumTupleType=EnumTupleType.INT, default: tuple[Any]=None, clip_min: Optional[int]=None, clip_max: Optional[int]=None, zero=0) -> tuple[List[Any]]:

    ret = []
    unified = data.get(key, [default])
    if not isinstance(unified, (list, tuple, set)):
        unified = [unified]

    for entry in unified:
        size = len(entry)
        newboi = []
        for idx in range(size):
            d = default[idx] if default is not None and idx < len(default) else None
            # entry could be a dict, list/tuple...
            v = entry
            if isinstance(entry, dict):
                v = entry.get(str(idx), d)
            elif isinstance(entry, (list, tuple, set)):
                v = entry[idx] if idx < len(entry) else d

            match typ:
                case EnumTupleType.FLOAT:
                    if isinstance(v, str):
                        parts = v.split('.', 1)
                        if len(parts) > 1:
                            v ='.'.join(parts[:2])
                    v = float(v if v is not None else zero)

                case EnumTupleType.LIST:
                    if v is not None:
                        v = v.split(',')

                case EnumTupleType.INT:
                    v = int(v if v is not None else zero)

            if typ in [EnumTupleType.INT, EnumTupleType.FLOAT]:
                if clip_min is not None:
                    v = max(v, clip_min)
                if clip_max is not None:
                    v = min(v, clip_max)

            if v == 0:
                v = zero
            newboi.append(v)

        ret.append(tuple(newboi))
    return ret

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
    iterators = [iter(iterable) for iterable in iterables]

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
