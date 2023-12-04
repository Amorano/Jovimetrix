"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural, Compositing & Video Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix
"""

import os
import sys
import math
from enum import Enum
from contextlib import contextmanager
from typing import Any, Generator

from PIL import Image
from PIL.PngImagePlugin import PngInfo

# =============================================================================
# === "LOGGER" ===
# =============================================================================
JOV_LOG = 0
try: JOV_LOG = int(os.getenv("JOV_LOG"))
except: pass

def logerr(msg: str, *arg) -> None:
    print(f"\033[48;2;135;27;81;93m[JOV]\033[0m {msg}", *arg)

def logwarn(msg: str, *arg) -> None:
    if JOV_LOG > 0:
        print(f"\033[48;2;189;135;54;93m[JOV]\033[0m {msg}", *arg)

def loginfo(msg: str, *arg) -> None:
    if JOV_LOG > 1:
        print(f"\033[48;2;54;135;27;93m[JOV]\033[0m {msg}", *arg)

def logdebug(msg: str, *arg) -> None:
    if JOV_LOG > 2:
        print(f"\033[48;2;35;87;181;93m[JOV]\033[0m {msg}", *arg)

@contextmanager
def suppress_std() -> Generator[None, Any, None]:
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull

        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# =============================================================================
# == CUSTOM ENUMERATION CLASS
# =============================================================================

class EnumJovian(Enum): pass

# =============================================================================

def zip_longest_fill(*iterables) -> Generator[tuple[Any | None, ...], Any, None]:
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
    """
    def _deep_merge(d1, d2) -> Any | dict:
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

def mergePNGMeta(root: str, target: str) -> None:
    for r, _, fs in os.walk(root):
        for f in fs:
            f, ext = os.path.splitext(f)
            if ext != '.json':
                continue

            img = f"{r}/{f}.png"
            if not os.path.isfile(img):
                continue

            fn = f"{r}/{f}.json"
            with open(fn, "r", encoding="utf-8") as out:
                data = out.read()

            out = f"{target}/{f}.png"
            with Image.open(img) as image:
                metadata = PngInfo()
                for i in image.text:
                    if i == 'workflow':
                        continue
                    metadata.add_text(i, str(image.text[i]))
                metadata.add_text("workflow", data.encode('utf-8'))
                image.save(out, pnginfo=metadata)
                loginfo(f"wrote {f} ==> {out}")

def gridMake(data: list[object]) -> list[object]:
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

if __name__ == "__main__":
    mergePNGMeta('../../pysssss-workflows', './flow')
