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

JOV_CATEGORY = "VARIABLE"

# ==============================================================================
# === CLASS ===
# ==============================================================================

class ValueNode(CozyBaseNode):
    NAME = "VALUE (JOV) 🧬"
    CATEGORY = JOV_CATEGORY
    RETURN_TYPES = (COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY, COZY_TYPE_ANY,)
    RETURN_NAMES = ("🦄", "X", "Y", "Z", "W",)
    SORT = 5
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
                "A": (COZY_TYPE_ANY, {
                    "default": None,
                    "tooltip":"Passes a raw value directly, or supplies defaults for any value inputs without connections"}),
                "TYPE": (typ, {
                    "default": EnumConvertType.BOOLEAN.name,
                    "tooltip":"Take the input and convert it into the selected type."}),
                "X": (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.maxsize, "maj": sys.maxsize,
                    "forceInput": True}),
                "Y": (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.maxsize, "maj": sys.maxsize,
                    "forceInput": True}),
                "Z": (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.maxsize, "maj": sys.maxsize,
                    "forceInput": True}),
                "W": (COZY_TYPE_NUMERICAL, {
                    "default": 0, "mij": -sys.maxsize, "maj": sys.maxsize,
                    "forceInput": True}),
                "AA": ("VEC4", {
                    "default": (0, 0, 0, 0), #"mij": -sys.maxsize, "maj": sys.maxsize,
                    "label": ["X", "Y"],
                    "tooltip":"default value vector for A"}),
                "BB": ("VEC4", {
                    "default": (1,1,1,1), #"mij": -sys.maxsize, "maj": sys.maxsize,
                    "label": ["X", "Y", "Z", "W"],
                    "tooltip":"default value vector for B"}),
                "SEED": ("INT", {
                    "default": 0, "min": 0, "max": sys.maxsize}),
            }
        })
        return d

    def run(self, **kw) -> tuple[bool]:
        raw = parse_param(kw, "A", EnumConvertType.ANY, 0)
        r_x = parse_param(kw, "X", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        r_y = parse_param(kw, "Y", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        r_z = parse_param(kw, "Z", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        r_w = parse_param(kw, "W", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        typ = parse_param(kw, "TYPE", EnumConvertType, EnumConvertType.BOOLEAN.name)
        xyzw = parse_param(kw, "AA", EnumConvertType.VEC4, (0, 0, 0, 0))
        seed = parse_param(kw, "SEED", EnumConvertType.INT, 0, 0)
        yyzw = parse_param(kw, "BB", EnumConvertType.VEC4, (1, 1, 1, 1))
        x_str = parse_param(kw, "STRING", EnumConvertType.STRING, "")
        params = list(zip_longest_fill(raw, r_x, r_y, r_z, r_w, typ, xyzw, seed, yyzw, x_str))
        results = []
        pbar = ProgressBar(len(params))
        old_seed = -1
        for idx, (raw, r_x, r_y, r_z, r_w, typ, xyzw, seed, yyzw, x_str) in enumerate(params):
            default = [x_str]
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
    RETURN_TYPES = ("VEC2",)
    RETURN_NAMES = ("VEC2",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Vector2 with float values",
    )
    SORT = 290
    DESCRIPTION = """
Outputs a VECTOR2.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "X": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "X channel value"}),
                "Y": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Y channel value"}),
                "A": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default X channel value"}),
                "B": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default Y channel value"}),
            }
        })
        return d

    def run(self, **kw) -> tuple[tuple[float, ...], tuple[int, ...]]:
        x = parse_param(kw, "X", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        y = parse_param(kw, "Y", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        a = parse_param(kw, "A", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        b = parse_param(kw, "B", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)

        result = []
        params = list(zip_longest_fill(x, y, a, b))
        pbar = ProgressBar(len(params))
        for idx, (x, y, a, b) in enumerate(params):
            x = round(a, 9) if x is None else round(x, 9)
            y = round(b, 9) if y is None else round(y, 9)
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
    SORT = 292
    DESCRIPTION = """
Outputs a VECTOR3.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "X": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "X channel value"}),
                "Y": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Y channel value"}),
                "Z": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Z channel value"}),
                "A": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default X channel value"}),
                "B": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default Y channel value"}),
                "C": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default Z channel value"}),
            }
        })
        return d

    def run(self, **kw) -> tuple[tuple[float, ...], tuple[int, ...]]:
        x = parse_param(kw, "X", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        y = parse_param(kw, "Y", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        z = parse_param(kw, "Z", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        a = parse_param(kw, "A", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        b = parse_param(kw, "B", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        c = parse_param(kw, "C", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        result = []
        params = list(zip_longest_fill(x, y, z, a, b, c))
        pbar = ProgressBar(len(params))
        for idx, (x, y, z, a, b, c) in enumerate(params):
            x = round(a, 9) if x is None else round(x, 9)
            y = round(b, 9) if y is None else round(y, 9)
            z = round(c, 9) if z is None else round(z, 9)
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
    SORT = 294
    DESCRIPTION = """
Outputs a VEC4.
"""

    @classmethod
    def INPUT_TYPES(cls) -> InputType:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                "X": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "X channel value"}),
                "Y": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Y channel value"}),
                "Z": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Z channel value"}),
                "W": (COZY_TYPE_NUMBER, {
                    "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "W channel value"}),
                "A": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default X channel value"}),
                "B": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default Y channel value"}),
                "C": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default Z channel value"}),
                "D": ("FLOAT", {
                    "default": 0, "min": -sys.maxsize, "max": sys.maxsize,
                    "tooltip": "Default W channel value"}),
            }
        })
        return d

    def run(self, **kw) -> tuple[tuple[float, ...], tuple[int, ...]]:
        x = parse_param(kw, "X", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        y = parse_param(kw, "Y", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        z = parse_param(kw, "Z", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        w = parse_param(kw, "W", EnumConvertType.FLOAT, None, -sys.maxsize, sys.maxsize)
        a = parse_param(kw, "A", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        b = parse_param(kw, "B", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        c = parse_param(kw, "C", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        d = parse_param(kw, "D", EnumConvertType.FLOAT, 0, -sys.maxsize, sys.maxsize)
        result = []
        params = list(zip_longest_fill(x, y, z, w, a, b, c, d))
        pbar = ProgressBar(len(params))
        for idx, (x, y, z, w, a, b, c, d) in enumerate(params):
            x = round(a, 9) if x is None else round(x, 9)
            y = round(b, 9) if y is None else round(y, 9)
            z = round(c, 9) if z is None else round(z, 9)
            w = round(d, 9) if w is None else round(w, 9)
            result.append((x, y, z, w,))
            pbar.update_absolute(idx)
        return result,

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
                "IN": (COZY_TYPE_ANY, {"default": None}),
            }
        })
        return d

    def run(self, ident, **kw) -> tuple[Any]:
        return kw["IN"],
'''