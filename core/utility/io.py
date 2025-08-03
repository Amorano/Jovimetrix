""" Jovimetrix - Utility """

import os
import json
from uuid import uuid4
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy.utils import ProgressBar
from folder_paths import get_output_directory
from nodes import interrupt_processing

from cozy_comfyui import \
    logger, \
    InputType, EnumConvertType, \
    deep_merge, parse_param, parse_param_list, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_IMAGE, COZY_TYPE_ANY, \
    CozyBaseNode

from cozy_comfyui.image.convert import \
    tensor_to_pil, tensor_to_cv

from cozy_comfyui.api import \
    TimedOutException, ComfyAPIMessage, \
    comfy_api_post

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

JOV_CATEGORY = "UTILITY/IO"

# min amount of time before showing the cancel dialog
JOV_DELAY_MIN = 5
try: JOV_DELAY_MIN = int(os.getenv("JOV_DELAY_MIN", JOV_DELAY_MIN))
except: pass
JOV_DELAY_MIN = max(1, JOV_DELAY_MIN)

# max 115 days
JOV_DELAY_MAX = 10000000
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
# === SUPPORT ===
# ==============================================================================

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

# ==============================================================================
# === CLASS ===
# ==============================================================================

class DelayNode(CozyBaseNode):
    NAME = "DELAY (JOV) ✋🏽"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY,)
    RETURN_NAMES = ("OUT",)
    OUTPUT_TOOLTIPS = (
        "Pass through data when the delay ends",
    )
    DESCRIPTION = """
Introduce pauses in the workflow that accept an optional input to pass through and a timer parameter to specify the duration of the delay. If no timer is provided, it defaults to a maximum delay. During the delay, it periodically checks for messages to interrupt the delay. Once the delay is completed, it returns the input passed to it. You can disable the screensaver with the `ENABLE` option
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PASS_IN: (COZY_TYPE_ANY, {
                    "default": None,
                    "tooltip":"The data that should be held until the timer completes."}),
                Lexicon.TIMER: ("INT", {
                    "default" : 0, "min": -1,
                    "tooltip":"How long to delay if enabled. 0 means no delay."}),
                Lexicon.ENABLE: ("BOOLEAN", {
                    "default": True,
                    "tooltip":"Enable or disable the screensaver."})
            }
        })
        return Lexicon._parse(d)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float('nan')

    def run(self, ident, **kw) -> tuple[Any]:
        delay = parse_param(kw, Lexicon.TIMER, EnumConvertType.INT, -1, -1, JOV_DELAY_MAX)[0]
        if delay < 0:
            delay = JOV_DELAY_MAX
        if delay > JOV_DELAY_MIN:
            comfy_api_post("jovi-delay-user", ident, {"id": ident, "timeout": delay})
        # enable = parse_param(kw, Lexicon.ENABLE, EnumConvertType.BOOLEAN, True)[0]

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

        return kw[Lexicon.PASS_IN]

class ExportNode(CozyBaseNode):
    NAME = "EXPORT (JOV) 📽"
    CATEGORY = JOV_CATEGORY
    NOT_IDEMPOTENT = True
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    DESCRIPTION = """
Responsible for saving images or animations to disk. It supports various output formats such as GIF and GIFSKI. Users can specify the output directory, filename prefix, image quality, frame rate, and other parameters. Additionally, it allows overwriting existing files or generating unique filenames to avoid conflicts. The node outputs the saved images or animation as a tensor.
"""

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float('nan')

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: (COZY_TYPE_IMAGE, {}),
                Lexicon.PATH: ("STRING", {
                    "default": get_output_directory(),
                    "default_top": "<comfy output dir>",}),
                Lexicon.FORMAT: (FORMATS, {
                    "default": FORMATS[0],}),
                Lexicon.PREFIX: ("STRING", {
                    "default": "jovi",}),
                Lexicon.OVERWRITE: ("BOOLEAN", {
                    "default": False,}),
                # GIF ONLY
                Lexicon.OPTIMIZE: ("BOOLEAN", {
                    "default": False,}),
                # GIFSKI ONLY
                Lexicon.QUALITY: ("INT", {
                    "default": 90, "min": 1, "max": 100,}),
                Lexicon.QUALITY_M: ("INT", {
                    "default": 100, "min": 1, "max": 100,}),
                # GIF OR GIFSKI
                Lexicon.FPS: ("INT", {
                    "default": 24, "min": 1, "max": 60,}),
                # GIF OR GIFSKI
                Lexicon.LOOP: ("INT", {
                    "default": 0, "min": 0,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> None:
        images = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        suffix = parse_param(kw, Lexicon.PREFIX, EnumConvertType.STRING, uuid4().hex[:16])[0]
        output_dir = parse_param(kw, Lexicon.PATH, EnumConvertType.STRING, "")[0]
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

        images = [tensor_to_pil(i) for i in images]
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

class RouteNode(CozyBaseNode):
    NAME = "ROUTE (JOV) 🚌"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("BUS",) + (COZY_TYPE_ANY,) * 10
    RETURN_NAMES = ("ROUTE",)
    OUTPUT_TOOLTIPS = (
        "Pass through for Route node",
    )
    DESCRIPTION = """
Routes the input data from the optional input ports to the output port, preserving the order of inputs. The `PASS_IN` optional input is directly passed through to the output, while other optional inputs are collected and returned as tuples, preserving the order of insertion.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.ROUTE: ("BUS", {
                    "default": None,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[Any, ...]:
        inout = parse_param(kw, Lexicon.ROUTE, EnumConvertType.ANY, None)
        vars = kw.copy()
        vars.pop(Lexicon.ROUTE, None)
        vars.pop('ident', None)

        parsed = []
        values = list(vars.values())
        for x in values:
            p = parse_param_list(x, EnumConvertType.ANY, None)
            parsed.append(p)
        return inout, *parsed,

class SaveOutputNode(CozyBaseNode):
    NAME = "SAVE OUTPUT (JOV) 💾"
    CATEGORY = JOV_CATEGORY
    NOT_IDEMPOTENT = True
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    DESCRIPTION = """
Save images with metadata to any specified path. Can save user metadata and prompt information.
"""

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float('nan')

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES(True, True)
        d = deep_merge(d, {
            "optional": {
                Lexicon.IMAGE: ("IMAGE", {}),
                Lexicon.PATH: ("STRING", {
                    "default": "", "dynamicPrompts":False}),
                Lexicon.NAME: ("STRING", {
                    "default": "output", "dynamicPrompts":False,}),
                Lexicon.META: ("JSON", {
                    "default": None,}),
                Lexicon.USER: ("STRING", {
                    "default": "", "multiline": True, "dynamicPrompts":False,}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> dict[str, Any]:
        image = parse_param(kw, Lexicon.IMAGE, EnumConvertType.IMAGE, None)
        path = parse_param(kw, Lexicon.PATH, EnumConvertType.STRING, "")
        fname = parse_param(kw, Lexicon.NAME, EnumConvertType.STRING, "output")
        metadata = parse_param(kw, Lexicon.META, EnumConvertType.DICT, {})
        usermeta = parse_param(kw, Lexicon.USER, EnumConvertType.DICT, {})
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
            image = tensor_to_cv(image)
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

            outname = fname
            if len(params) > 1:
                outname += f"_{idx}"
            outname = (root / outname).with_suffix(".png")
            logger.info(f"wrote file: {outname}")
            image.save(outname, pnginfo=meta_png)
            pbar.update_absolute(idx)
        return ()
