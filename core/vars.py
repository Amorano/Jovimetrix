""" Jovimetrix - Variables """

import sys
import random

from comfy.utils import ProgressBar

from cozy_comfyui import \
    InputType, EnumConvertType, \
    deep_merge, parse_param, parse_value, zip_longest_fill

from cozy_comfyui.node import \
    COZY_TYPE_ANY, COZY_TYPE_NUMERICAL, COZY_TYPE_NUMBER, \
    CozyBaseNode

from .. import \
    Lexicon

JOV_CATEGORY = "VARIABLE"

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ValueNode(CozyBaseNode):
    NAME = "VALUE (JOV) 🧬"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY,)
    RETURN_NAMES = (Lexicon.ANY_OUT, Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W)
    SORT = 5
    DESCRIPTION = """
Supplies raw or default values for various data types, supporting vector input with components for X, Y, Z, and W. It also provides a string input option.
"""
    UPDATE = False

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()

        typ = EnumConvertType._member_names_
        for t in ['IMAGE', 'LATENT', 'ANY', 'MASK', 'LAYER']:
            try: typ.pop(typ.index(t))
            except: pass

        d = deep_merge(d, {
            "optional": {
                Lexicon.IN_A: (COZY_TYPE_ANY, {
                    "default": None,
                    "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
                Lexicon.TYPE: (typ, {
                    "default": EnumConvertType.BOOLEAN.name,
                    "tooltip":"Take the input and convert it into the selected type."}),
                Lexicon.X: (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.maxsize, "maj": sys.maxsize,
                    "forceInput": True}),
                Lexicon.Y: (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.maxsize, "maj": sys.maxsize,
                    "forceInput": True}),
                Lexicon.Z: (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.maxsize, "maj": sys.maxsize,
                    "forceInput": True}),
                Lexicon.W: (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.maxsize, "maj": sys.maxsize,
                    "forceInput": True}),
                Lexicon.IN_A+Lexicon.IN_A: ("VEC4", {
                    "default": (0, 0, 0, 0), #"mij": -sys.maxsize, "maj": sys.maxsize,
                    "label": [Lexicon.X, Lexicon.Y],
                    "tooltip":"default value vector for A"}),
                "SEED": ("INT", {"default": 0, "min": 0, "max": sys.maxsize}),
                Lexicon.IN_B+Lexicon.IN_B: ("VEC4", {
                    "default": (1,1,1,1), #"mij": -sys.maxsize, "maj": sys.maxsize,
                    "label": [Lexicon.X, Lexicon.Y, Lexicon.Z, Lexicon.W],
                    "tooltip":"default value vector for B"}),
                Lexicon.STRING: ("STRING", {
                    "default": "", "dynamicPrompts": False, "multiline": True}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[bool]:
        raw = parse_param(kw, Lexicon.IN_A, EnumConvertType.ANY, [0])
        r_x = parse_param(kw, Lexicon.X, EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        r_y = parse_param(kw, Lexicon.Y, EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        r_z = parse_param(kw, Lexicon.Z, EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        r_w = parse_param(kw, Lexicon.W, EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        typ = parse_param(kw, Lexicon.TYPE, EnumConvertType, EnumConvertType.BOOLEAN.name)
        xyzw = parse_param(kw, Lexicon.IN_A+Lexicon.IN_A, EnumConvertType.VEC4, [(0, 0, 0, 0)])
        seed = parse_param(kw, "SEED", EnumConvertType.INT, 0, 0)
        yyzw = parse_param(kw, Lexicon.IN_B+Lexicon.IN_B, EnumConvertType.VEC4, [(1, 1, 1, 1)])
        x_str = parse_param(kw, Lexicon.STRING, EnumConvertType.STRING, "")
        params = list(zip_longest_fill(raw, r_x, r_y, r_z, r_w, typ, xyzw, seed, yyzw, x_str))
        results = []
        pbar = ProgressBar(len(params))
        old_seed = -1
        for idx, (raw, r_x, r_y, r_z, r_w, typ, xyzw, seed, yyzw, x_str) in enumerate(params):
            default = [x_str]
            default2 = None
            if typ not in [EnumConvertType.STRING, EnumConvertType.LIST, \
                        EnumConvertType.DICT,\
                        EnumConvertType.IMAGE, EnumConvertType.LATENT, \
                        EnumConvertType.ANY, EnumConvertType.MASK]:
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
            if seed != 0 and isinstance(val, (tuple, list,)) and isinstance(val2, (tuple, list,)):
                self.UPDATE = True
                # mutable to update
                val = list(val)
                for i in range(len(val)):
                    mx = max(val[i], val2[i])
                    mn = min(val[i], val2[i])
                    if mn == mx:
                        val[i] = mn
                    else:
                        if old_seed != seed:
                            random.seed(seed)
                            old_seed = seed
                        if typ in [EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4]:
                            val[i] = mn + random.random() * (mx - mn)
                        else:
                            val[i] = random.randint(mn, mx)

            out = parse_value(val, typ, val) or 0.
            items = [0.,0.,0.,0.]
            if not isinstance(out, (list, tuple,)):
                items[0] = out
            else:
                for i in range(len(out)):
                    items[i] = out[i]
            results.append([out, *items])
            pbar.update_absolute(idx)
        if len(results) < 2:
            return results[0]
        return *list(zip(*results)),

class Vector2Node(CozyBaseNode):
    NAME = "VECTOR2 (JOV)"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("VEC2", "VEC2INT", )
    RETURN_NAMES = ("VEC2", "VEC2INT", )
    OUTPUT_TOOLTIPS = (
        "Vector2 with float values",
        "Vector2 with integer values",
    )
    SORT = 290
    DESCRIPTION = """
Outputs a VEC2 or VEC2INT.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "X": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "1st channel value"}),
                "Y": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "2nd channel value"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[tuple[float, ...], tuple[int, ...]]:
        x = parse_param(kw, "X", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        y = parse_param(kw, "Y", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        results = []
        params = list(zip_longest_fill(x, y))
        pbar = ProgressBar(len(params))
        for idx, (x, y) in enumerate(params):
            x = round(x, 6)
            y = round(y, 6)
            results.append([(x, y,), (int(x), int(y),)])
            pbar.update_absolute(idx)
        return *list(zip(*results)),

class Vector3Node(CozyBaseNode):
    NAME = "VECTOR3 (JOV)"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("VEC3", "VEC3INT", )
    RETURN_NAMES = ("VEC3", "VEC3INT", )
    OUTPUT_TOOLTIPS = (
        "Vector3 with float values",
        "Vector3 with integer values",
    )
    SORT = 292
    DESCRIPTION = """
Outputs a VEC3 or VEC3INT.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "X": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "1st channel value"}),
                "Y": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "2nd channel value"}),
                "Z": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "3rd channel value"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[tuple[float, ...], tuple[int, ...]]:
        x = parse_param(kw, "X", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        y = parse_param(kw, "Y", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        z = parse_param(kw, "Z", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        results = []
        params = list(zip_longest_fill(x, y, z))
        pbar = ProgressBar(len(params))
        for idx, (x, y, z) in enumerate(params):
            x = round(x, 6)
            y = round(y, 6)
            z = round(z, 6)
            results.append([(x, y, z,), (int(x), int(y), int(z),)])
            pbar.update_absolute(idx)
        return *list(zip(*results)),

class Vector4Node(CozyBaseNode):
    NAME = "VECTOR4 (JOV)"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ("VEC4", "VEC4INT", )
    RETURN_NAMES = ("VEC4", "VEC4INT", )
    OUTPUT_TOOLTIPS = (
        "Vector4 with float values",
        "Vector4 with integer values",
    )
    SORT = 294
    DESCRIPTION = """
Outputs a VEC4 or VEC4INT.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "X": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "1st channel value"}),
                "Y": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "2nd channel value"}),
                "Z": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "3rd channel value"}),
                "W": (COZY_TYPE_NUMBER, {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "4th channel value"}),
            }
        })
        return Lexicon._parse(d)

    def run(self, **kw) -> tuple[tuple[float, ...], tuple[int, ...]]:
        x = parse_param(kw, "X", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        y = parse_param(kw, "Y", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        z = parse_param(kw, "Z", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        w = parse_param(kw, "W", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        results = []
        params = list(zip_longest_fill(x, y, z, w))
        pbar = ProgressBar(len(params))
        for idx, (x, y, z, w,) in enumerate(params):
            x = round(x, 6)
            y = round(y, 6)
            z = round(z, 6)
            w = round(w, 6)
            results.append([(x, y, z, w,), (int(x), int(y), int(z), int(w),)])
            pbar.update_absolute(idx)
        return *list(zip(*results)),

'''
class ParameterNode(CozyBaseNode):
    NAME = "PARAMETER (JOV) ⚙️"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    SORT = 100
    DESCRIPTION = """

"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.PASS_IN: (COZY_TYPE_ANY, {"default": None}),
            }
        })
        return Lexicon._parse(d)

    def run(self, ident, **kw) -> tuple[Any]:
        return kw[Lexicon.PASS_IN],
'''