"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Utility
"""

import os
import json
from uuid import uuid4
from pathlib import Path
from typing import Any, Tuple

import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from loguru import logger

from comfy.utils import ProgressBar
from folder_paths import get_output_directory
from nodes import interrupt_processing

from Jovimetrix import JOV_TYPE_ANY, JOV_TYPE_IMAGE, Lexicon, JOVBaseNode, \
    ComfyAPIMessage, TimedOutException, DynamicInputType, \
    comfy_message, deep_merge

from Jovimetrix.sup.util import EnumConvertType, path_next, parse_param, \
    zip_longest_fill

from Jovimetrix.sup.image import tensor2cv, tensor2pil

# ==============================================================================

JOV_CATEGORY = "UTILITY"

# min amount of time before showing the cancel dialog
JOV_DELAY_MIN = 5
try: JOV_DELAY_MIN = int(os.getenv("JOV_DELAY_MIN", JOV_DELAY_MIN))
except: pass
JOV_DELAY_MIN = max(1, JOV_DELAY_MIN)

# max 10 minutes to start
JOV_DELAY_MAX = 600
try: JOV_DELAY_MAX = int(os.getenv("JOV_DELAY_MAX", JOV_DELAY_MAX))
except: pass

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

# ==============================================================================

class DelayNode(JOVBaseNode):
    NAME = "DELAY (JOV) ✋🏽"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = (JOV_TYPE_ANY,)
    RETURN_NAMES = (Lexicon.PASS_OUT,)
    SORT = 240
    DESCRIPTION = """
Introduce pauses in the workflow that accept an optional input to pass through and a timer parameter to specify the duration of the delay. If no timer is provided, it defaults to a maximum delay. During the delay, it periodically checks for messages to interrupt the delay. Once the delay is completed, it returns the input passed to it. You can disable the screensaver with the `ENABLE` option
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PASS_IN: (JOV_TYPE_ANY, {"default": None}),
                Lexicon.TIMER: ("INT", {"default" : 0, "mij": -1}),
                Lexicon.ENABLE: ("BOOLEAN", {"default": True, "tooltips":"Enable or disable the screensaver"})
            },
            "outputs": {
                0: (Lexicon.PASS_OUT, {"tooltips":"Pass through data when the delay ends"})
            }
        })
        return Lexicon._parse(d, cls)

    def run(self, ident, **kw) -> Tuple[Any]:
        delay = parse_param(kw, Lexicon.TIMER, EnumConvertType.INT, -1, 0, JOV_DELAY_MAX)[0]
        if delay < 0:
            delay = JOV_DELAY_MAX
        if delay > JOV_DELAY_MIN:
            comfy_message(ident, "jovi-delay-user", {"id": ident, "timeout": delay})
        # enable = parse_param(kw, Lexicon.ENABLE, EnumConvertType.BOOLEAN, True)

        step = 1
        pbar = ProgressBar(delay)
        while step <= delay:
            try:
                data = ComfyAPIMessage.poll(ident, timeout=1)
                if data.get('id', None) == ident:
                    if data.get('cmd', False) == False:
                        interrupt_processing(True)
                        logger.warning(f"delay [cancelled] ({step}): {ident}")
                    break
            except TimedOutException as _:
                if step % 10 == 0:
                    logger.info(f"delay [continue] ({step}): {ident}")
            pbar.update_absolute(step)
            step += 1
        return kw[Lexicon.PASS_IN],

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

class RouteNode(JOVBaseNode):
    NAME = "ROUTE (JOV) 🚌"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    RETURN_TYPES = ("BUS",) + (JOV_TYPE_ANY,) * 127
    RETURN_NAMES = (Lexicon.ROUTE,)
    SORT = 850
    DESCRIPTION = """
Routes the input data from the optional input ports to the output port, preserving the order of inputs. The `PASS_IN` optional input is directly passed through to the output, while other optional inputs are collected and returned as tuples, preserving the order of insertion.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        e = {
            "optional": {
                Lexicon.ROUTE: ("BUS", {"default": None, "tooltips":"Pass through another route node to pre-populate the outputs."}),
            },
            "outputs": {
                0: (Lexicon.ROUTE, {"tooltips":"Pass through for Route node"})
            }
        }
        d = deep_merge(d, e)
        return Lexicon._parse(d, cls)

    def run(self, **kw) -> Tuple[Any, ...]:
        inout = parse_param(kw, Lexicon.ROUTE, EnumConvertType.ANY, [None])
        vars = kw.copy()
        vars.pop(Lexicon.ROUTE, None)
        vars.pop('ident', None)
        return inout, *vars.values(),

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
