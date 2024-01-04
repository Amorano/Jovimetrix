"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

              Animation, Image Compositing & Procedural Creation
                    http://www.github.com/amorano/jovimetrix

                    Copyright 2023 Alexander Morano (Joviex)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

GO NUTS; JUST TRY NOT TO DO IT IN YOUR HEAD.

@title: Jovimetrix
@category: Compositing
@tags: adjust, animate, audio, compose, compositing, composition, device, flow,
video, mask, shape, webcam, audio, animation, logic
@description: Webcams, GLSL shader, Media Streaming, Tick animation, Image manipulation,
Polygonal shapes, MIDI, MP3/WAVE, Flow Logic
@author: amorano
@reference: https://github.com/Amorano/Jovimetrix
@node list:
    AdjustNode, ColorMatchNode, FindEdgeNode, HSVNode, LevelsNode, ThresholdNode,
    TickNode, WaveGeneratorNode,
    GraphWaveNode,
    ConversionNode, CalcUnaryOPNode, CalcBinaryOPNode, ValueNode
    TransformNode, BlendNode, PixelSplitNode, PixelMergeNode, MergeNode, CropNode, ColorTheoryNode,
    ConstantNode, ShapeNode, TextNode, GLSLNode,
    StreamReaderNode, StreamWriterNode, MIDIMessageNode, MIDIReaderNode, MIDIFilterEZNode, MIDIFilterNode,
    ComparisonNode, IfThenElseNode, GetNode, SetNode,
    OptionsNode, AkashicNode, ValueGraphNode
@version: 0.999999999
"""

import os
import json
import shutil
import inspect
import importlib
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Tuple, Union

import torch
import numpy as np

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

ROOT = Path(__file__).resolve().parent
ROOT_COMFY = ROOT.parent.parent
ROOT_COMFY_WEB = ROOT_COMFY / "web" / "extensions" / "jovimetrix"

JOV_CONFIG = {}
JOV_WEB = ROOT / 'web'
JOV_DEFAULT = JOV_WEB / 'default.json'
JOV_CONFIG_FILE = JOV_WEB / 'config.json'

JOV_MAX_DELAY = 60.
try: JOV_MAX_DELAY = float(os.getenv("JOV_MAX_DELAY", 60.))
except: pass

# =============================================================================
# === TYPE SHORTCUTS ===
# =============================================================================

TYPE_COORD = Union[
    tuple[int, int],
    tuple[float, float]
]

TYPE_PIXEL = Union[
    int,
    float,
    Tuple[float, float, float],
    Tuple[float, float, float, Optional[float]],
    Tuple[int, int, int],
    Tuple[int, int, int, Optional[int]]
]

TYPE_IMAGE = Union[np.ndarray, torch.Tensor]

# =============================================================================
# === THERE CAN BE ONLY ONE ===
# =============================================================================

class Singleton(type):
    _instances = {}

    def __call__(cls, *arg, **kw) -> Any:
        # If the instance does not exist, create and store it
        if cls not in cls._instances:
            instance = super().__call__(*arg, **kw)
            cls._instances[cls] = instance
        return cls._instances[cls]

# =============================================================================
# === LOGGER ===
# =============================================================================

class Logger(metaclass=Singleton):
    _LEVEL = int(os.getenv("JOV_LOG_LEVEL", 0))

    @classmethod
    def _raw(cls, color, who, *arg) -> None:
        if who is None:
            print(color, '[JOV]\033[0m', *arg)
        else:
            print(color, '[JOV]\033[0m', f'({who})', *arg)

    @classmethod
    def dump(cls, *arg) -> None:
        who = inspect.currentframe().f_back.f_code.co_name
        cls._raw("\033[48;2;35;127;81;93m", who, None, *arg)

    @classmethod
    def err(cls, *arg) -> None:
        who = inspect.currentframe().f_back.f_code.co_name
        cls._raw("\033[48;2;135;27;81;93m", who, None, *arg)

    @classmethod
    def warn(cls, *arg) -> None:
        if Logger._LEVEL > 0:
           # who = inspect.currentframe().f_back.f_code.co_name
            cls._raw("\033[48;2;159;155;44;93m", None, *arg)

    @classmethod
    def info(cls, *arg) -> None:
        if Logger._LEVEL > 1:
            cls._raw("\033[48;2;44;115;37;93m", None, *arg)

    @classmethod
    def debug(cls, *arg) -> None:
        if Logger._LEVEL > 2:
            t = datetime.now().strftime('%H:%M:%S.%f')
            who = inspect.currentframe().f_back.f_code.co_name
            cls._raw("\033[48;2;35;87;181;93m", t, who, *arg)

    @classmethod
    def spam(cls, *arg) -> None:
        if Logger._LEVEL > 3:
            t = datetime.now().strftime('%H:%M:%S.%f')
            who = inspect.currentframe().f_back.f_code.co_name
            cls._raw("\033[48;2;35;87;181;93m", t, who, *arg)

# =============================================================================
# === SESSION ===
# =============================================================================

def configLoad() -> None:
    global JOV_CONFIG
    try:
        with open(JOV_CONFIG_FILE, 'r', encoding='utf-8') as fn:
            JOV_CONFIG = json.load(fn)
    except (IOError, FileNotFoundError) as e:
        pass
    except Exception as e:
        Logger.err(e)

class Session(metaclass=Singleton):
    CLASS_MAPPINGS = {}
    CLASS_MAPPINGS_WIP = {}

    @classmethod
    def ignore_files(cls, d, files) -> list[str]|None:
        return [x for x in files if x.endswith('.json') or x.endswith('.html')]

    def __init__(self, *arg, **kw) -> None:
        global JOV_CONFIG
        found = False
        if JOV_CONFIG_FILE.exists():
            configLoad()
            # is this an old config, copy default (sorry, not sorry)
            found = JOV_CONFIG.get('user', None) is not None

        if not found:
            try:
                shutil.copy2(JOV_DEFAULT, JOV_CONFIG_FILE)
                Logger.warn("---> DEFAULT CONFIGURATION <---")
            except:
                raise Exception("MAJOR 😿😰😬🥟 BLUNDERCATS 🥟😬😰😿")

        for f in (ROOT / 'nodes').iterdir():
            if f.suffix != ".py" or f.stem.startswith('_'):
                continue

            module = importlib.import_module(f"Jovimetrix.nodes.{f.stem}")
            classes = inspect.getmembers(module, inspect.isclass)
            for class_name, class_object in classes:
                # assume both attrs are good enough....
                if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                    name = class_object.NAME
                    if hasattr(class_object, 'POST'):
                        class_object.CATEGORY = "JOVIMETRIX 🔺🟩🔵/WIP ☣️💣"
                        Session.CLASS_MAPPINGS_WIP[name] = class_object
                    else:
                        Session.CLASS_MAPPINGS[name] = class_object

            Logger.info("✅", module.__name__)

        # 🔗 ⚓ 📀 🍿 🎪 🐘 🤯 😱 💀 ⛓️ 🔒 🔑 🪀 🪁 🔮 🧿 🧙🏽 🧙🏽‍♀️ 🧯 🦚

        NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in Session.CLASS_MAPPINGS.items()}
        Session.CLASS_MAPPINGS.update({k: v for k, v in Session.CLASS_MAPPINGS_WIP.items()})

        NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k, _ in Session.CLASS_MAPPINGS_WIP.items()})

        Session.CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(Session.CLASS_MAPPINGS.items(),
                                                              key=lambda item: getattr(item[1], 'SORT', 0))}
        # now sort the categories...
        for c in ["CREATE", "ADJUST", "TRANSFORM", "COMPOSE",
                  "ANIMATE", "FLOW", "CALC", "DEVICE", "AUDIO",
                  "UTILITY", "WIP ☣️💣"]:

            prime = Session.CLASS_MAPPINGS.copy()
            for k, v in prime.items():
                if v.CATEGORY.endswith(c):
                    NODE_CLASS_MAPPINGS[k] = v
                    Session.CLASS_MAPPINGS.pop(k)
                    Logger.debug("✅", k)

        # anything we dont know about sort last...
        for k, v in Session.CLASS_MAPPINGS.items():
            NODE_CLASS_MAPPINGS[k] = v
            Logger.debug('⁉️', k, v)

session = Session()
