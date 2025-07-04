""" Jovimetrix - Variables """

import sys
import random
from typing import Any

from comfy.utils import ProgressBar

from cozy_comfyui import \
    InputType, EnumConvertType, \
    deep_merge, parse_param, parse_value, zip_longest_fill

from cozy_comfyui.lexicon import \
    Lexicon

from cozy_comfyui.node import \
    COZY_TYPE_ANY, COZY_TYPE_NUMERICAL, \
    CozyBaseNode

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

JOV_CATEGORY = "VARIABLE"

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ValueNode(CozyBaseNode):
    NAME = "VALUE (JOV) 🧬"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY,)
    RETURN_NAMES = ("❔", Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W,)
    OUTPUT_IS_LIST = (True, True, True, True, True,)
    DESCRIPTION = """
Supplies raw or default values for various data types, supporting vector input with components for X, Y, Z, and W. It also provides a string input option.
"""
    UPDATE = False

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        typ = EnumConvertType._member_names_[:6]
        d = deep_merge(d, {
            "optional": {
                Lexicon.IN_A: (COZY_TYPE_ANY, {
                    "default": None,}),
                Lexicon.X: (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "forceInput": True}),
                Lexicon.Y: (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "forceInput": True}),
                Lexicon.Z: (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "forceInput": True}),
                Lexicon.W: (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "forceInput": True}),
                Lexicon.TYPE: (typ, {
                    "default": EnumConvertType.BOOLEAN.name}),
                Lexicon.DEFAULT_A: ("VEC4", {
                    "default": (0, 0, 0, 0), "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W]}),
                Lexicon.DEFAULT_B: ("VEC4", {
                    "default": (1,1,1,1), "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W]}),
                Lexicon.SEED: ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[tuple[Any, ...]]:
        raw = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, 0)
        r_x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, None)
        r_y = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, None)
        r_z = parse_param(kw, Lexicon.Z, EnumConvertType.FLOAT, None)
        r_w = parse_param(kw, Lexicon.W, EnumConvertType.FLOAT, None)
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType, EnumConvertType.BOOLEAN.name)
        xyzw = parse_param(kw, Lexicon.DEFAULT_A, EnumConvertType.VEC4, (0, 0, 0, 0))
        yyzw = parse_param(kw, Lexicon.DEFAULT_B, EnumConvertType.VEC4, (1, 1, 1, 1))
        seed = parse_param(kw, Lexicon.SEED, EnumConvertType.INT, 0)
        params = list(zip_longest_fill(raw, r_x, r_y, r_z, r_w, typ, xyzw, yyzw, seed))
        results = []
        pbar = ProgressBar(len(params))
        old_seed = -1
        for idx, (raw, r_x, r_y, r_z, r_w, typ, xyzw, yyzw, seed) in enumerate(params):
            # default = [x_str]
            default2 = None
            a, b, c, d = xyzw
            a2, b2, c2, d2 = yyzw
            default = (a if r_x is None else r_x,
                b if r_y is None else r_y,
                c if r_z is None else r_z,
                d if r_w is None else r_w)
            default2 = (a2, b2, c2, d2)

            val = parse_value(raw, typ, default)
            val2 = parse_value(default2, typ, default2)

            # check if set to randomize....
            self.UPDATE = False
            if seed != 0:
                self.UPDATE = True
                val = list(val) if isinstance(val, (tuple, list,)) else [val]
                val2 = list(val2) if isinstance(val2, (tuple, list,)) else [val2]

                for i in range(len(val)):
                    mx = max(val[i], val2[i])
                    mn = min(val[i], val2[i])
                    if mn == mx:
                        val[i] = mn
                    else:
                        if old_seed != seed:
                            random.seed(seed)
                            old_seed = seed
                        if typ in [EnumConvertType.INT, EnumConvertType.BOOLEAN]:
                            val[i] = random.randint(mn, mx)
                        else:
                            val[i] = mn + random.random() * (mx - mn)

            out = parse_value(val, typ, val)
            items = [out,0,0,0] if not isinstance(out, (tuple, list,)) else out
            results.append([out, *items])
            pbar.update_absolute(idx)

        return *list(zip(*results)),

class Vector2Node(CozyBaseNode):
    NAME = "VECTOR2 (JOV)"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("VEC2",)
    RETURN_NAMES = ("VEC2",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Vector2 with float values",
    )
    DESCRIPTION = """
Outputs a VECTOR2.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.X: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "X channel value"}),
                Lexicon.Y: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "Y channel value"}),
                Lexicon.DEFAULT: ("VEC2", {
                    "default": (0,0), "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "tooltip": "Default vector value"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[tuple[float, ...]]:
        x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, None)
        y = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, None)
        default = parse_param(kw, Lexicon.DEFAULT, EnumConvertType.VEC2, (0,0))
        result = []
        params = list(zip_longest_fill(x, y, default))
        pbar = ProgressBar(len(params))
        for idx, (x, y, default) in enumerate(params):
            x = round(default[0], 9) if x is None else round(x, 9)
            y = round(default[1], 9) if y is None else round(y, 9)
            result.append((x, y,))
            pbar.update_absolute(idx)
        return result,

class Vector3Node(CozyBaseNode):
    NAME = "VECTOR3 (JOV)"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("VEC3",)
    RETURN_NAMES = ("VEC3",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Vector3 with float values",
    )
    DESCRIPTION = """
Outputs a VECTOR3.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.X: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "X channel value"}),
                Lexicon.Y: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "Y channel value"}),
                Lexicon.Z: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "Z channel value"}),
                Lexicon.DEFAULT: ("VEC3", {
                    "default": (0,0,0), "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "tooltip": "Default vector value"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[tuple[float, ...]]:
        x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, None)
        y = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, None)
        z = parse_param(kw, Lexicon.Z, EnumConvertType.FLOAT, None)
        default = parse_param(kw, Lexicon.DEFAULT, EnumConvertType.VEC3, (0,0,0))
        result = []
        params = list(zip_longest_fill(x, y, z, default))
        pbar = ProgressBar(len(params))
        for idx, (x, y, z, default) in enumerate(params):
            x = round(default[0], 9) if x is None else round(x, 9)
            y = round(default[1], 9) if y is None else round(y, 9)
            z = round(default[2], 9) if z is None else round(z, 9)
            result.append((x, y, z,))
            pbar.update_absolute(idx)
        return result,

class Vector4Node(CozyBaseNode):
    NAME = "VECTOR4 (JOV)"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("VEC4",)
    RETURN_NAMES = ("VEC4",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Vector4 with float values",
    )
    DESCRIPTION = """
Outputs a VECTOR4.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.X: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "X channel value"}),
                Lexicon.Y: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "Y channel value"}),
                Lexicon.Z: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "Z channel value"}),
                Lexicon.W: (COZY_TYPE_NUMERICAL, {
                    "min": -sys.float_info.max, "max": sys.float_info.max,
                    "tooltip": "W channel value"}),
                Lexicon.DEFAULT: ("VEC4", {
                    "default": (0,0,0,0), "mij": -sys.float_info.max, "maj": sys.float_info.max,
                    "tooltip": "Default vector value"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[tuple[float, ...]]:
        x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, None)
        y = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, None)
        z = parse_param(kw, Lexicon.Z, EnumConvertType.FLOAT, None)
        w = parse_param(kw, Lexicon.W, EnumConvertType.FLOAT, None)
        default = parse_param(kw, Lexicon.DEFAULT, EnumConvertType.VEC4, (0,0,0,0))
        result = []
        params = list(zip_longest_fill(x, y, z, w, default))
        pbar = ProgressBar(len(params))
        for idx, (x, y, z, w, default) in enumerate(params):
            x = round(default[0], 9) if x is None else round(x, 9)
            y = round(default[1], 9) if y is None else round(y, 9)
            z = round(default[2], 9) if z is None else round(z, 9)
            w = round(default[3], 9) if w is None else round(w, 9)
            result.append((x, y, z, w,))
            pbar.update_absolute(idx)
        return result,
