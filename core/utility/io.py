"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import os
import json
from uuid import uuid4
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from loguru import logger

from comfy.utils import ProgressBar
from folder_paths import get_output_directory

from Jovimetrix import JOV_TYPE_ANY, JOV_TYPE_IMAGE, \
    Lexicon, JOVBaseNode, deep_merge

from Jovimetrix.sup.util import EnumConvertType, \
    path_next, parse_param, zip_longest_fill

from Jovimetrix.sup.image import tensor2cv, tensor2pil

# =============================================================================

JOV_CATEGORY = "UTILITY"

FORMATS = ["gif", "png", "jpg"]
if (JOV_GIFSKI := os.getenv("JOV_GIFSKI", None)) is not None:
    if not os.path.isfile(JOV_GIFSKI):
        logger.error(f"gifski missing [{JOV_GIFSKI}]")
        JOV_GIFSKI = None
    else:
        FORMATS = ["gifski"] + FORMATS
        logger.info("gifski support")
else:
    logger.warning("no gifski support")

# =============================================================================

class ExportNode(JOVBaseNode):
    NAME = "EXPORT (JOV) 📽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    SORT = 2000
    DESCRIPTION = """
Responsible for saving images or animations to disk. It supports various output formats such as GIF and GIFSKI. Users can specify the output directory, filename prefix, image quality, frame rate, and other parameters. Additionally, it allows overwriting existing files or generating unique filenames to avoid conflicts. The node outputs the saved images or animation as a tensor.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PIXEL: (JOV_TYPE_IMAGE, {}),
                Lexicon.PASS_OUT: ("STRING", {"default": get_output_directory(), "default_top":"<comfy output dir>"}),
                Lexicon.FORMAT: (FORMATS, {"default": FORMATS[0]}),
                Lexicon.PREFIX: ("STRING", {"default": "jovi"}),
                Lexicon.OVERWRITE: ("BOOLEAN", {"default": False}),
                # GIF ONLY
                Lexicon.OPTIMIZE: ("BOOLEAN", {"default": False}),
                # GIFSKI ONLY
                Lexicon.QUALITY: ("INT", {"default": 90, "mij": 1, "maj": 100}),
                Lexicon.QUALITY_M: ("INT", {"default": 100, "mij": 1, "maj": 100}),
                # GIF OR GIFSKI
                Lexicon.FPS: ("INT", {"default": 24, "mij": 1, "maj": 60}),
                # GIF OR GIFSKI
                Lexicon.LOOP: ("INT", {"default": 0, "mij": 0}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> None:
        images = parse_param(kw, Lexicon.PIXEL, EnumConvertType.IMAGE, None)
        suffix = parse_param(kw, Lexicon.PREFIX, EnumConvertType.STRING, uuid4().hex[:16])[0]
        output_dir = parse_param(kw, Lexicon.PASS_OUT, EnumConvertType.STRING, "")[0]
        format = parse_param(kw, Lexicon.FORMAT, EnumConvertType.STRING, "gif")[0]
        overwrite = parse_param(kw, Lexicon.OVERWRITE, EnumConvertType.BOOLEAN, False)[0]
        optimize = parse_param(kw, Lexicon.OPTIMIZE, EnumConvertType.BOOLEAN, False)[0]
        quality = parse_param(kw, Lexicon.QUALITY, EnumConvertType.INT, 90, 0, 100)[0]
        motion = parse_param(kw, Lexicon.QUALITY_M, EnumConvertType.INT, 100, 0, 100)[0]
        fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1, 60)[0]
        loop = parse_param(kw, Lexicon.LOOP, EnumConvertType.INT, 0, 0)[0]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def output(extension) -> Path:
            path = output_dir / f"{suffix}.{extension}"
            if not overwrite and os.path.isfile(path):
                path = str(output_dir / f"{suffix}_%s.{extension}")
                path = path_next(path)
            return path

        images = [tensor2pil(i) for i in images]
        if format == "gifski":
            root = output_dir / f"{suffix}_{uuid4().hex[:16]}"
            # logger.debug(root)
            try:
                root.mkdir(parents=True, exist_ok=True)
                for idx, i in enumerate(images):
                    fname = str(root / f"{suffix}_{idx}.png")
                    i.save(fname)
            except Exception as e:
                logger.warning(output_dir)
                logger.error(str(e))
                return
            else:
                out = output('gif')
                fps = f"--fps {fps}" if fps > 0 else ""
                q = f"--quality {quality}"
                mq = f"--motion-quality {motion}"
                cmd = f"{JOV_GIFSKI} -o {out} {q} {mq} {fps} {str(root)}/{suffix}_*.png"
                logger.info(cmd)
                try:
                    os.system(cmd)
                except Exception as e:
                    logger.warning(cmd)
                    logger.error(str(e))

                # shutil.rmtree(root)

        elif format == "gif":
            images[0].save(
                output('gif'),
                append_images=images[1:],
                disposal=2,
                duration=1 / fps * 1000 if fps else 0,
                loop=loop,
                optimize=optimize,
                save_all=True,
            )
        else:
            for img in images:
                img.save(output(format), optimize=optimize)
        return ()

class SaveOutput(JOVBaseNode):
    NAME = "SAVE OUTPUT (JOV) 💾"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    SORT = 85
    DESCRIPTION = """
Save the output image along with its metadata to the specified path. Supports saving additional user metadata and prompt information.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES(True, True)
        d = deep_merge(d, {
            "optional": {
                "image": ("IMAGE",),
                "path": ("STRING", {"default": "", "dynamicPrompts":False}),
                "fname": ("STRING", {"default": "output", "dynamicPrompts":False}),
                "metadata": ("JSON", {}),
                "usermeta": ("STRING", {"multiline": True, "dynamicPrompts":False,
                                        "default": ""}),
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> dict[str, Any]:
        image = parse_param(kw, 'image', EnumConvertType.IMAGE, None)
        metadata = parse_param(kw, 'metadata', EnumConvertType.DICT, {})
        usermeta = parse_param(kw, 'usermeta', EnumConvertType.DICT, {})
        path = parse_param(kw, 'path', EnumConvertType.STRING, "")
        fname = parse_param(kw, 'fname', EnumConvertType.STRING, "output")
        prompt = parse_param(kw, 'prompt', EnumConvertType.STRING, "")
        pnginfo = parse_param(kw, 'extra_pnginfo', EnumConvertType.DICT, {})
        params = list(zip_longest_fill(image, path, fname, metadata, usermeta, prompt, pnginfo))
        pbar = ProgressBar(len(params))
        for idx, (image, path, fname, metadata, usermeta, prompt, pnginfo) in enumerate(params):
            if image is None:
                logger.warning("no image")
                image = torch.zeros((32, 32, 4), dtype=torch.uint8, device="cpu")
            try:
                if not isinstance(usermeta, (dict,)):
                    usermeta = json.loads(usermeta)
                metadata.update(usermeta)
            except json.decoder.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(e)
                logger.error(usermeta)
            metadata["prompt"] = prompt
            metadata["workflow"] = json.dumps(pnginfo)
            image = tensor2cv(image)
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            meta_png = PngInfo()
            for x in metadata:
                try:
                    data = json.dumps(metadata[x])
                    meta_png.add_text(x, data)
                except Exception as e:
                    logger.error(e)
                    logger.error(x)
            if path == "" or path is None:
                path = get_output_directory()
            root = Path(path)
            if not root.exists():
                root = Path(get_output_directory())
            root.mkdir(parents=True, exist_ok=True)
            fname = (root / fname).with_suffix(".png")
            logger.info(f"wrote file: {fname}")
            image.save(fname, pnginfo=meta_png)
            pbar.update_absolute(idx)
        return ()
