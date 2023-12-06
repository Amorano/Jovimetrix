"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
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

@title: Jovimetrix Composition Pack
@category: Compositing
@tags: compositing, composition, video, mask, shape, webcam
@description: Procedural & Compositing. Includes a Webcam node.
@author: amorano
@reference: https://github.com/Amorano/Jovimetrix
@node list: ConstantNode, ShapeNode, PixelShaderNode, PixelShaderImageNode,
            TransformNode, TileNode, MirrorNode, ExtendNode, HSVNode, AdjustNode,
            BlendNode, ThresholdNode, ProjectionNode, StreamReadNode, StreamWriteNode,
            RouteNode, TickNode, OptionsNode
@version: 0.98
"""

import inspect
import importlib
from pathlib import Path
from typing import Any

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "web"

#
#
#

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs) -> Any:
        # If the instance does not exist, create and store it
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

# =============================================================================
# === COMFYUI NODE MAP ===
# =============================================================================
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

class Session(metaclass=Singleton):
    CLASS_MAPPINGS = {}
    CLASS_MAPPINGS_WIP = {}

    def __init__(self, *arg, **kw) -> None:
        root = Path(__file__).parent.absolute() / 'nodes'
        for f in root.iterdir():
            if f.suffix != ".py" or f.stem.startswith('_'):
                continue

            module = importlib.import_module(f"Jovimetrix.nodes.{f.stem}")
            classes = inspect.getmembers(module, inspect.isclass)
            for class_name, class_object in classes:
                # assume both attrs are good enough....
                if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME') and hasattr(class_object, 'CATEGORY'):
                    name = class_object.NAME
                    if hasattr(class_object, 'POST'):
                        class_object.CATEGORY = "JOVIMETRIX 🔺🟩🔵/💣☣️ WIP ☣️💣"
                        Session.CLASS_MAPPINGS_WIP[name] = class_object
                    else:
                        Session.CLASS_MAPPINGS[name] = class_object

            print("✅", module.__name__)

        # 🔗 ⚓ 📀 🍿 🎪 🐘 🤯 😱 💀 ⛓️ 🔒 🔑 🪀 🪁 🔮 🧿 🧙🏽 🧙🏽‍♀️ 🧯 🦚

        NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, _ in Session.CLASS_MAPPINGS.items()}
        Session.CLASS_MAPPINGS.update({k: v for k, v in Session.CLASS_MAPPINGS_WIP.items()})

        NODE_DISPLAY_NAME_MAPPINGS.update({k: k for k, _ in Session.CLASS_MAPPINGS_WIP.items()})

        Session.CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(Session.CLASS_MAPPINGS.items(),
                                                        key=lambda item: getattr(item[1], 'SORT', 0))}
        # NODE_CLASS_MAPPINGS = {}

        # now sort the categories...
        for c in ["CREATE", "ADJUST", "TRANSFORM", "COMPOSE", "ANIMATE", "AUDIO", "DEVICE", "UTILITY", "💣☣️ WIP ☣️💣"]:

            prime = Session.CLASS_MAPPINGS.copy()
            for k, v in prime.items():
                if v.CATEGORY.endswith(c):
                    NODE_CLASS_MAPPINGS[k] = v
                    Session.CLASS_MAPPINGS.pop(k)
                    print('✅', k, v)

        # anything we dont know about sort last...
        for k, v in Session.CLASS_MAPPINGS.items():
            NODE_CLASS_MAPPINGS[k] = v
            print('⁉️', k, v)

session = Session()

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
