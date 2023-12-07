
import os

from Jovimetrix import Logger

from PIL.PngImagePlugin import PngInfo
from PIL import Image

# =============================================================================
# === Jovimetrix Cleanup Support ===
# =============================================================================

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
                Logger.info(f"{f} ==> {out}")

# =============================================================================
# === ASILE 5 ===
# =============================================================================

if __name__ == "__main__":
    mergePNGMeta('../../pysssss-workflows', './flow')