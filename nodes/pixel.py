"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Pixel manipulation -- Images
"""

import os
import shutil
from typing import Any
from pathlib import Path
from uuid import uuid4

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from loguru import logger

import comfy
from folder_paths import get_output_directory
# from server import PromptServer
import nodes

from Jovimetrix import path_next, JOVBaseNode, \
    IT_REQUIRED, IT_PIXEL, MIN_IMAGE_SIZE, IT_PIXEL2

from Jovimetrix.sup.lexicon import Lexicon
from Jovimetrix.sup.util import deep_merge_dict, zip_longest_fill
from Jovimetrix.sup.image import cv2mask, cv2tensor, image_diff, tensor2cv, tensor2pil

FORMATS = ["gif", "png", "jpg"]
if (JOV_GIFSKI := os.getenv("JOV_GIFSKI", None)) is not None:
    FORMATS = ["gifski"] + FORMATS
    logger.info("gifski support")
else:
    logger.warning("no gifski support")

class ExportNode(JOVBaseNode):
    NAME = "EXPORT (JOV) 📽"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/IMAGE"
    DESCRIPTION = ""
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    SORT = 10

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"optional": {
            Lexicon.PREFIX: ("STRING", {"default": ""}),
            Lexicon.PASS_OUT: ("STRING", {"default": get_output_directory()}),
            Lexicon.OVERWRITE: ("BOOLEAN", {"default": False}),
            Lexicon.FORMAT: (FORMATS, {"default": FORMATS[0]}),
            # GIF ONLY
            Lexicon.OPTIMIZE: ("BOOLEAN", {"default": False}),
            # GIFSKI ONLY
            Lexicon.QUALITY: ("INT", {"default": 90, "min": 1, "max": 100}),
            Lexicon.QUALITY_M: ("INT", {"default": 100, "min": 1, "max": 100}),
            # GIF OR GIFSKI
            Lexicon.FPS: ("INT", {"default": 20, "min": 1, "max": 60}),
            # GIF OR GIFSKI
            Lexicon.LOOP: ("INT", {"default": 0, "min": 0}),
        }}
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL, d)

    def run(self, **kw) -> None:
        img = kw.get(Lexicon.PIXEL, [None])[0]
        suffix = kw.get(Lexicon.PREFIX, [""])[0]
        if suffix == "":
            suffix = uuid4().hex[:16]

        output_dir = kw.get(Lexicon.PASS_OUT, [""])[0]
        format = kw.get(Lexicon.FORMAT, ["gif"])[0]
        overwrite = kw.get(Lexicon.OVERWRITE, False)[0]
        optimize = kw.get(Lexicon.OPTIMIZE, [False])[0]
        quality = kw.get(Lexicon.QUALITY, [0])[0]
        motion = kw.get(Lexicon.QUALITY_M, [0])[0]
        fps = kw.get(Lexicon.FPS, [0])[0]
        loop = kw.get(Lexicon.LOOP, [0])[0]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def output(extension) -> Path:
            path = output_dir / f"{suffix}.{extension}"
            if not overwrite and os.path.isfile(path):
                path = str(output_dir / f"{suffix}_%s.{extension}")
                path = path_next(path)
            return path

        empty = Image.new("RGB", (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE))
        images = [tensor2pil(i) if i is not None else empty for i in img]
        images = [i.convert("RGB") if i is not None else empty for i in images]

        if format == "gifski":
            root = output_dir / f"{suffix}_{uuid4().hex[:16]}"
            try:
                root.mkdir(parents=True, exist_ok=True)
                for idx, i in enumerate(images):
                    fname = str(root / f"{suffix}_{idx}.png")
                    i.save(fname)
            except Exception as e:
                logger.warning(output_dir)
                logger.error(str(e))
                return

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

            shutil.rmtree(root)

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

class ImageDiffNode(JOVBaseNode):
    NAME = "IMAGE DIFF (JOV) 📏"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/IMAGE"
    DESCRIPTION = "Explicitly show the differences between two images."
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, True, True, )
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "FLOAT", )
    RETURN_NAMES = (Lexicon.IN_A, Lexicon.IN_B, Lexicon.DIFF, Lexicon.THRESHOLD, Lexicon.FLOAT, )
    SORT = 30

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return deep_merge_dict(IT_REQUIRED, IT_PIXEL2)

    def run(self, **kw) -> tuple[Any, Any]:
        a = kw.get(Lexicon.PIXEL_A, [None])
        b = kw.get(Lexicon.PIXEL_B, [None])
        image_a = []
        image_b = []
        diff = []
        thresh = []
        score = []
        params = [tuple(x) for x in zip_longest_fill(a, b)]
        if len(params) == 0:
            e = [cv2tensor(np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8))]
            m = [cv2mask(np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=np.uint8))]
            return e, e, m, m, [1.],

        pbar = comfy.utils.ProgressBar(len(params))
        for idx, (a, b) in enumerate(params):
            if a is None and b is None:
                e = cv2tensor(np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, 3), dtype=np.uint8))
                image_a.append(e)
                image_b.append(e)
                m = cv2mask(np.zeros((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), dtype=np.uint8))
                diff.append(m)
                thresh.append(m)
                score.append(1.)
                continue

            width = MIN_IMAGE_SIZE
            height = MIN_IMAGE_SIZE
            _, height, width, _ = a.shape if a is not None else b.shape
            a = tensor2cv(a) if a is not None else np.zeros((height, width, 3), dtype=np.uint8)
            b = tensor2cv(b) if b is not None else np.zeros((height, width, 3), dtype=np.uint8)

            a, b, d, t, s = image_diff(a, b)
            image_a.append(cv2tensor(a))
            image_b.append(cv2tensor(b))
            diff.append(cv2mask(d))
            thresh.append(cv2mask(t))
            score.append(s)
            pbar.update_absolute(idx)

        return image_a, image_b, diff, thresh, score,
