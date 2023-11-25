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
import math
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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
            with open(fn, "r", encoding="utf8") as out:
                data = out.read()

            out = f"{target}/{f}.png"
            with Image.open(img) as image:
                metadata = PngInfo()
                for i in image.text:
                    if i == 'workflow':
                        continue
                    metadata.add_text(i, str(image.text[i]))
                metadata.add_text("workflow", data)
                image.save(out, pnginfo=metadata)
                print(f"wrote {f} ==> {img}")

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
    mergePNGMeta('../../../pysssss-workflows', '../flow')
