"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

              Animation, Image Compositing & Procedural Creation
                    http://www.github.com/amorano/jovimetrix

@title: Jovimetrix
@author: amorano
@category: Compositing
@reference: https://github.com/Amorano/Jovimetrix
@tags: adjust, animate, compose, compositing, composition, device, flow, video,
mask, shape, webcam, animation, logic
@description: Integrates Webcam, MIDI, Spout and GLSL shader support. Animation
via tick. Parameter manipulation with wave generator. Math operations with Unary
and Binary support. Value converstion for all major types (int, string, list,
dict, Image, Mask). Shape mask generation, image stacking and channel ops, batch
splitting, merging and randomizing, load images and video from anywhere, dynamic
bus routing with a single node, export support for GIPHY, save output anywhere!
flatten, crop, transform; check colorblindness, make stereogram or stereoscopic
images, or liner interpolate values and more.
@node list:
    ConstantNode, GLSLNode, ShapeNode, StereogramNode, StereoscopicNode, TextNode, WaveGraphNode,
    AdjustNode, ColorBlindNode, ColorMatchNode, FilterMaskNode, ThresholdNode,
    BlendNode, ColorTheoryNode, CropNode, FlattenNode, PixelMergeNode, PixelSplitNode,
    PixelSwapNode, StackNode, TransformNode
    CalcUnaryOPNode, CalcBinaryOPNode, ValueNode, ConvertNode, LerpNode, DelayNode,
    ComparisonNode,
    TickNode, WaveGeneratorNode,
    MIDIMessageNode, MIDIReaderNode, MIDIFilterEZNode, MIDIFilterNode,
    StreamReaderNode, StreamWriterNode, SpoutWriter,
    AkashicNode, ArrayNode, BatchLoadNode, DynamicNode, ValueGraphNode, ExportNode, QueueNode,
    RouteNode, SaveOutputNode
@version: 1.2.6
"""

import os
import sys
import time
import json
import shutil
import inspect
import importlib
from pathlib import Path
from typing import Any

try:
    from server import PromptServer
    from aiohttp import web
except:
    pass

from loguru import logger

NODE_LIST_MAP = {}
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

ROOT = Path(__file__).resolve().parent
ROOT_COMFY = ROOT.parent.parent

JOV_CONFIG = {}
JOV_WEB = ROOT / 'web'
JOV_DEFAULT = JOV_WEB / 'default.json'
JOV_CONFIG_FILE = JOV_WEB / 'config.json'

# nodes to skip on import; for online systems; skip Export, Streamreader, etc...
JOV_IGNORE_NODE = ROOT / 'ignore.txt'
JOV_SIDECAR = os.getenv("JOV_SIDECAR", str(ROOT / "_md"))

JOV_LOG_LEVEL = os.getenv("JOV_LOG_LEVEL", "WARNING")
logger.configure(handlers=[{"sink": sys.stdout, "level": JOV_LOG_LEVEL}])

JOV_INTERNAL = os.getenv("JOV_INTERNAL", 'false').strip().lower() in ('true', '1', 't')

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
    RETURN_TYPES = ()
    FUNCTION = "run"
    # instance map for caching
    INSTANCE = {}

    @classmethod
    def INPUT_TYPES(cls, prompt:bool=False, extra_png:bool=False) -> dict:
        data = {
            "required": {},
            "hidden": {
                "ident": "UNIQUE_ID"
            }
        }
        if prompt:
            data["hidden"]["prompt"] = "PROMPT"
        if extra_png:
            data["hidden"]["extra_pnginfo"] = "EXTRA_PNGINFO"
        return data

# wildcard trick is 100% stolen from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

JOV_TYPE_ANY = AnyType("*")

# want to make explicit entries; comfy only looks for single type
JOV_TYPE_COMFY = "BOOLEAN,FLOAT,INT"
JOV_TYPE_VECTOR = "VEC2,VEC3,VEC4,VEC2INT,VEC3INT,VEC4INT,COORD2D"
JOV_TYPE_NUMBER = f"{JOV_TYPE_COMFY},{JOV_TYPE_VECTOR}"
JOV_TYPE_IMAGE = "IMAGE,MASK"
JOV_TYPE_FULL = f"{JOV_TYPE_NUMBER},{JOV_TYPE_IMAGE}"

JOV_TYPE_COMFY = JOV_TYPE_ANY
JOV_TYPE_VECTOR = JOV_TYPE_ANY
JOV_TYPE_NUMBER = JOV_TYPE_ANY
JOV_TYPE_IMAGE = JOV_TYPE_ANY
JOV_TYPE_FULL =JOV_TYPE_ANY

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
        if len(JOV_CONFIG) == 0:
            JOV_CONFIG = configLoad(JOV_CONFIG_FILE)
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

    @PromptServer.instance.routes.get("/jovimetrix/doc")
    async def jovimetrix_doc(request) -> Any:
        from Jovimetrix.sup.lexicon import get_node_info, json2markdown
        data = {}
        global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        for k, v in NODE_CLASS_MAPPINGS.items():
            display_name = NODE_DISPLAY_NAME_MAPPINGS[k]
            ret = {"class": v, "display_name": display_name}
            data[k] = get_node_info(ret)
            data[k]['.md'] = json2markdown(data[k])
            fname = display_name.split(" (JOV)")[0]
            path = Path(JOV_SIDECAR.replace("{name}", fname))
            path.mkdir(parents=True, exist_ok=True)
            with open(str(path / f"{fname}.md"), "w", encoding='utf-8') as f:
                f.write(data[k]['.md'])
        return web.json_response(data)

except Exception as e:
    logger.error(e)

# =============================================================================
# == SUPPORT FUNCTIONS
# =============================================================================

def parse_reset(ident:str) -> int:
    try:
        data = ComfyAPIMessage.poll(ident, timeout=0)
        return data.get('cmd', None) == 'reset'
    except TimedOutException as e:
        return -1
    except Exception as e:
        logger.error(str(e))

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
        global JOV_CONFIG, JOV_IGNORE_NODE, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, NODE_LIST_MAP
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

        if JOV_IGNORE_NODE.exists():
            JOV_IGNORE_NODE = configLoad(JOV_IGNORE_NODE, False)
        else:
            JOV_IGNORE_NODE = []

        node_count = 0
        for f in (ROOT / 'core').iterdir():
            if f.suffix != ".py" or f.stem.startswith('_'):
                continue
            if f.stem in JOV_IGNORE_NODE or f.stem+'.py' in JOV_IGNORE_NODE:
                logger.warning(f"💀 [IGNORED] Jovimetrix.core.{f.stem}")
                continue
            try:
                module = importlib.import_module(f"Jovimetrix.core.{f.stem}")
            except Exception as e:
                logger.warning(f"module failed {f}")
                logger.warning(str(e))
                continue

            # check if there is a dynamic register function....
            try:
                for class_name, class_def in module.import_dynamic():
                    setattr(module, class_name, class_def)
                    logger.info(f"shader: {class_name}")
            except Exception as e:
                pass

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

                    desc = class_object.DESCRIPTION if hasattr(class_object, 'DESCRIPTION') else ""
                    NODE_LIST_MAP[name] = desc.split('.')[0].strip('\n')
                    node_count += 1

            logger.info(f"✅ {module.__name__}")
        logger.info(f"{node_count} nodes loaded")

        NODE_DISPLAY_NAME_MAPPINGS = {k: v.NAME_PRETTY if hasattr(v, 'NAME_PRETTY') else k for k, v in Session.CLASS_MAPPINGS.items()}
        Session.CLASS_MAPPINGS.update({k: v for k, v in Session.CLASS_MAPPINGS_WIP.items()})
        NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k in Session.CLASS_MAPPINGS_WIP.keys()})
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
                    logger.debug(f"✅ {k} :: {NODE_DISPLAY_NAME_MAPPINGS[k]}")

        # anything we dont know about sort last...
        for k, v in Session.CLASS_MAPPINGS.items():
            NODE_CLASS_MAPPINGS[k] = v
            # logger.debug('⁉️ {} {}', k, v)

        # only do the list on local runs...
        if JOV_INTERNAL:
            with open(str(ROOT) + "/node_list.json", "w", encoding="utf-8") as f:
                json.dump(NODE_LIST_MAP, f, sort_keys=True, indent=4 )

session = Session()
