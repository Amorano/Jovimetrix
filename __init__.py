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
    AdjustNode, ColorMatchNode, ThresholdNode, ColorBlindNode,
    TickNode, WaveGeneratorNode,
    LoadWaveNode, GraphWaveNode,
    CalcUnaryOPNode, CalcBinaryOPNode, ValueNode, ConvertNode, LerpNode,
    TransformNode, BlendNode, PixelSplitNode, PixelMergeNode, PixelSwapNode,
    StackNode, CropNode, ColorTheoryNode,
    ConstantNode, ShapeNode, TextNode, StereogramNode, GLSLNode, NoiseNode,
    StreamReaderNode, StreamWriterNode, MIDIMessageNode, MIDIReaderNode,
    MIDIFilterEZNode, MIDIFilterNode, AudioDeviceNode,
    DelayNode, HoldValueNode, ComparisonNode, SelectNode,
    AkashicNode, ValueGraphNode, RouteNode, QueueNode, ExportNode, ImageDiffNode,
    BatchNode
@version: 0.9999999999999
"""

import os
import sys
import time
import json
import shutil
import inspect
import importlib
from pathlib import Path
from typing import Any, Optional, Tuple, Union

try:
    from server import PromptServer
    from aiohttp import web
except:
    pass

import torch
import numpy as np
from loguru import logger

from Jovimetrix.sup.lexicon import Lexicon

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
JOV_HELP_INDEX = {}
JOV_HELP_ROOT = ROOT / 'help'
# nodes to skip on import; for online systems; skip Export, Streamreader, etc...
JOV_IGNORE_NODE = ROOT / 'ignore.txt'
JOV_GLSL = ROOT / 'res' / 'glsl'

JOV_WEBWIKI_URL = "https://github.com/Amorano/Jovimetrix/wiki"
JOV_WEBHELP_ROOT = "https://github.com/Amorano/Jovimetrix-examples/blob/master"
JOV_WEBRES_ROOT = "https://raw.githubusercontent.com/Amorano/Jovimetrix-examples/master"

JOV_LOG_LEVEL = os.getenv("JOV_LOG_LEVEL", "WARNING")
logger.configure(handlers=[{"sink": sys.stdout, "level": JOV_LOG_LEVEL}])

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
TYPE_VECTOR = Union[TYPE_IMAGE|TYPE_PIXEL]

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
# === CORE NODES ===
# =============================================================================

class JOVBaseNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {}}
    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = False
    FUNCTION = "run"

class JOVImageSimple(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = (Lexicon.IMAGE,)

class JOVImageMultiple(JOVBaseNode):
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = (Lexicon.IMAGE, Lexicon.RGB, Lexicon.MASK,)

# wildcard trick is 100% stolen from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

WILDCARD = AnyType("*")

# =============================================================================
# == API RESPONSE
# =============================================================================

class TimedOutException(Exception): pass

class ComfyAPIMessage:
    # STASH = {}
    MESSAGE = {}

    #@classmethod
    #def send(cls, ident, message) -> None:
        #cls.MESSAGE[str(ident)] = message

    @classmethod
    def poll(cls, ident, period=0.01, timeout=3) -> Any:
        _t = time.monotonic()
        if isinstance(ident, (set, list, tuple, )):
            ident = ident[0]
        sid = str(ident)
        while not (sid in cls.MESSAGE) and time.monotonic() - _t < timeout:
            time.sleep(period)

        if not (sid in cls.MESSAGE):
            # logger.warning(f"message failed {sid}")
            raise TimedOutException
        dat = cls.MESSAGE.pop(sid)
        return dat

def comfy_message(ident:str, route:str, data:dict) -> None:
    data['id'] = ident
    PromptServer.instance.send_sync(route, data)

try:
    @PromptServer.instance.routes.post("/jovimetrix/message")
    async def jovimetrix_message(request) -> Any:
        json_data = await request.json()
        did = json_data.get("id", None)
        ComfyAPIMessage.MESSAGE[str(did)] = json_data
        # logger.debug(ComfyAPIMessage.MESSAGE[did])
        return web.json_response()

    @PromptServer.instance.routes.get("/jovimetrix/config")
    async def jovimetrix_config(request) -> Any:
        global JOV_CONFIG, JOV_CONFIG_FILE
        configLoad(JOV_CONFIG_FILE)
        return web.json_response(JOV_CONFIG)

    @PromptServer.instance.routes.post("/jovimetrix/config")
    async def jovimetrix_config_post(request) -> Any:
        json_data = await request.json()
        did = json_data.get("id", None)
        value = json_data.get("v", None)
        if did is None or value is None:
            logger.error("bad config {}", json_data)
            return

        global JOV_CONFIG
        from Jovimetrix.sup.util import update_nested_dict
        update_nested_dict(JOV_CONFIG, did, value)
        # logger.debug("{} {}", did, value)
        with open(JOV_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(JOV_CONFIG, f, indent=4)
        return web.json_response(json_data)

    @PromptServer.instance.routes.post("/jovimetrix/config/clear")
    async def jovimetrix_config_post(request) -> Any:
        json_data = await request.json()
        name = json_data['name']
        # logger.debug(name)
        global JOV_CONFIG
        try:
            del JOV_CONFIG['color'][name]
        except KeyError as _:
            pass
        with open(JOV_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(JOV_CONFIG, f)
        return web.json_response(json_data)

except Exception as e:
    logger.error(e)

# =============================================================================
# == SUPPORT FUNCTIONS
# =============================================================================

def load_help(name:str, category:str, desc:str, url:str) -> str:
    global JOV_HELP_INDEX, JOV_WEBWIKI_URL, JOV_WEBHELP_ROOT, JOV_WEBRES_ROOT
    parse = JOV_HELP_INDEX.get(name, "NO HELP AVAILABLE")
    parse = parse.replace("!NAME!", name).replace("!DESC!", desc).replace("!CAT!", category)
    parse = parse.replace("!URL!", f"[{name}]({JOV_WEBWIKI_URL}/{url})")
    name_raw = name.split('(JOV)')[0].lower().strip().replace(' ', '_')
    name_vid = f"![]({JOV_WEBRES_ROOT}/node/{name}/{name_raw}.gif)"
    name_vid = name_vid.replace(' ', '%20')
    print(name_vid)
    # https://raw.githubusercontent.com/Amorano/Jovimetrix-examples/master/node/BLEND%20(JOV)%20%E2%9A%97%EF%B8%8F/blend.gif
    # https://github.com/Amorano/Jovimetrix-examples/blob/master/node/BLEND%20(JOV)%20%E2%9A%97%EF%B8%8F/blend.gif
    parse = parse.replace("!URL_VID!", name_vid)
    return parse

def parse_reset(ident:str) -> int:
    try:
        data = ComfyAPIMessage.poll(ident, timeout=0)
        return data.get('cmd', None) == 'reset'
    except TimedOutException as e:
        return -1
    except Exception as e:
        logger.error(str(e))

# =============================================================================
# === GLOBALS ===
# =============================================================================

MIN_IMAGE_SIZE = 512

# =============================================================================
# === SESSION ===
# =============================================================================

def configLoad(fname:Path, as_json:bool=True) -> Any | list[str] | None:
    try:
        with open(fname, 'r', encoding='utf-8') as fn:
            if as_json:
                return json.load(fn)
            return fn.read().splitlines()
    except (IOError, FileNotFoundError) as e:
        pass
    except Exception as e:
        logger.error(e)
    return []

class Session(metaclass=Singleton):
    CLASS_MAPPINGS = {}
    CLASS_MAPPINGS_WIP = {}

    @classmethod
    def ignore_files(cls, d, files) -> list[str]|None:
        return [x for x in files if x.endswith('.json') or x.endswith('.html')]

    def __init__(self, *arg, **kw) -> None:
        global JOV_CONFIG, JOV_IGNORE_NODE
        found = False
        if JOV_CONFIG_FILE.exists():
            JOV_CONFIG = configLoad(JOV_CONFIG_FILE)
            # is this an old config, copy default (sorry, not sorry)
            found = JOV_CONFIG.get('user', None) is not None

        if not found:
            try:
                shutil.copy2(JOV_DEFAULT, JOV_CONFIG_FILE)
                logger.warning("---> DEFAULT CONFIGURATION <---")
            except:
                raise Exception("MAJOR 😿😰😬🥟 BLUNDERCATS 🥟😬😰😿")

        help_count = 0
        footer = "help system powered by [MelMass](https://github.com/melMass) and the [comfy_mtb](https://github.com/melMass/comfy_mtb) project"
        global JOV_HELP_ROOT, JOV_HELP_INDEX
        for f in (JOV_HELP_ROOT).iterdir():
            if f.suffix != ".md":
                continue
            if len(data := configLoad(f, as_json=False)) > 0:
                JOV_HELP_INDEX[f.stem] = '\n'.join(data)
            else:
                JOV_HELP_INDEX[f.stem] = """
!NAME! || !CAT!

!DESC!

WIKI: !URL!

!URL_VID!
"""
            JOV_HELP_INDEX[f.stem] = JOV_HELP_INDEX[f.stem] + f'\n\n{footer}'
            help_count += 1
        if help_count > 0:
            logger.info(f"{help_count} help files loaded")

        if JOV_IGNORE_NODE.exists():
            JOV_IGNORE_NODE = configLoad(JOV_IGNORE_NODE, False)
        else:
            JOV_IGNORE_NODE = []

        node_count = 0
        for f in (ROOT / 'core').iterdir():
            if f.suffix != ".py" or f.stem.startswith('_'):
                continue
            if f.stem in JOV_IGNORE_NODE or f.stem+'.py' in JOV_IGNORE_NODE:
                logger.warning(f"💀 Jovimetrix.core.{f.stem}")
                continue
            module = importlib.import_module(f"Jovimetrix.core.{f.stem}")
            try:
                module = importlib.import_module(f"Jovimetrix.core.{f.stem}")
            except Exception as e:
                logger.warning(f"module failed {f}")
                logger.warning(str(e))
                continue

            classes = inspect.getmembers(module, inspect.isclass)
            for class_name, class_object in classes:
                # assume both attrs are good enough....
                if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                    if (name := class_object.NAME) in JOV_IGNORE_NODE:
                        logger.warning(f"😥 {name}")
                        continue

                    if hasattr(class_object, 'POST'):
                        class_object.CATEGORY = "JOVIMETRIX 🔺🟩🔵/WIP ☣️💣"
                        Session.CLASS_MAPPINGS_WIP[name] = class_object
                    else:
                        Session.CLASS_MAPPINGS[name] = class_object

                    if JOV_HELP_INDEX.get(name, None) is None:
                        if hasattr(class_object, 'DESCRIPTION'):
                            JOV_HELP_INDEX[name] = class_object.DESCRIPTION
                        else:
                            JOV_HELP_INDEX[name] = "NO HELP AVAILABLE"
                            logger.debug(f"{name} missing help")

                    node_count += 1

            logger.info(f"✅ {module.__name__}")
        logger.info(f"{node_count} nodes loaded")

        # 🔗 ⚓ 📀 🍿 🎪 🐘 🤯 😱 💀 ⛓️ 🔒 🔑 🪀 🪁 🔮 🧿 🧙🏽 🧙🏽‍♀️ 🧯 🦚

        NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in Session.CLASS_MAPPINGS.items()}
        Session.CLASS_MAPPINGS.update({k: v for k, v in Session.CLASS_MAPPINGS_WIP.items()})

        NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k, _ in Session.CLASS_MAPPINGS_WIP.items()})

        Session.CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(Session.CLASS_MAPPINGS.items(),
                                                              key=lambda item: getattr(item[1], 'SORT', 0))}
        # now sort the categories...
        for c in ["CREATE", "ADJUST", "COMPOSE", "IMAGE",
                  "CALC", "ANIMATE", "FLOW", "DEVICE", "AUDIO",
                  "UTILITY", "WIP ☣️💣"]:

            prime = Session.CLASS_MAPPINGS.copy()
            for k, v in prime.items():
                if v.CATEGORY.endswith(c):
                    NODE_CLASS_MAPPINGS[k] = v
                    Session.CLASS_MAPPINGS.pop(k)
                    # logger.debug("✅ {}", k)

        # anything we dont know about sort last...
        for k, v in Session.CLASS_MAPPINGS.items():
            NODE_CLASS_MAPPINGS[k] = v
            # logger.debug('⁉️ {} {}', k, v)

session = Session()
