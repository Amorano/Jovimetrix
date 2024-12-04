"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import sys
from pathlib import Path
from typing import Any, Tuple

import torch
from loguru import logger


try:
    from server import PromptServer
    from aiohttp import web
except:
    pass
from comfy.utils import ProgressBar

from Jovimetrix import JOV_TYPE_IMAGE, Lexicon, JOVImageNode, \
    comfy_message, deep_merge

from Jovimetrix.sup.util import EnumConvertType, parse_param, \
    parse_value

from Jovimetrix.sup.image.adjust import EnumInterpolation, EnumScaleMode, \
    image_scalefit

from Jovimetrix.sup.image import MIN_IMAGE_SIZE, image_convert, tensor2cv, \
    cv2tensor_full

import Jovimetrix.sup.shader as glsl_enums

from Jovimetrix.sup.shader import JOV_ROOT_GLSL, GLSL_PROGRAMS, PROG_FRAGMENT, \
    PROG_VERTEX, PTYPE, CompileException, EnumGLSLEdge, GLSLShader, shader_meta, \
    load_file_glsl

# ==============================================================================

JOV_CATEGORY = "CREATE"

# ==============================================================================

try:
    @PromptServer.instance.routes.get("/jovimetrix/glsl")
    async def jovimetrix_glsl_list(request) -> Any:
        ret = {k:[kk for kk, vv in v.items() \
                  if kk not in ['NONE'] and vv not in [None] and Path(vv).exists()]
               for k, v in GLSL_PROGRAMS.items()}
        return web.json_response(ret)

    @PromptServer.instance.routes.get("/jovimetrix/glsl/{prog}/{shader}")
    async def jovimetrix_glsl_raw(request) -> Any:
        prog = request.match_info["prog"]
        if (program := GLSL_PROGRAMS.get(prog, None)) is None:
            return web.Response(text=f"no program {prog}")

        shader = request.match_info["shader"].replace("|", "/")
        if (raw := program.get(shader, None)) is None:
            return web.Response(text=f"no shader {shader}")

        response = load_file_glsl(raw)
        return web.Response(text=response)

    @PromptServer.instance.routes.post("/jovimetrix/glsl")
    async def jovimetrix_glsl(request) -> Any:
        json_data = await request.json()
        response = {k:None for k in json_data.keys()}
        for who in response.keys():
            if (programs := GLSL_PROGRAMS.get(who, None)) is None:
                logger.warning(f"no program type {who}")
                continue
            fname = json_data[who]
            if (data := programs.get(fname, None)) is not None:
                response[who] = load_file_glsl(data)
            else:
                logger.warning(f"no glsl shader entry {fname}")

        return web.json_response(response)
except Exception as e:
    logger.error(e)

class GLSLNodeBase(JOVImageNode):
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/GLSL"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d = deep_merge(d, {
            "optional": {
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.MATTE.name}),
                Lexicon.WH: ("VEC2INT", {"default": (512, 512), "mij":MIN_IMAGE_SIZE, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4INT", {"default": (0, 0, 0, 255), "rgb": True}),
                Lexicon.EDGE_X: (EnumGLSLEdge._member_names_, {"default": EnumGLSLEdge.CLAMP.name}),
                Lexicon.EDGE_Y: (EnumGLSLEdge._member_names_, {"default": EnumGLSLEdge.CLAMP.name}),
            }
        })
        return Lexicon._parse(d, cls)

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__glsl = GLSLShader()
        self.__delta = 0

    def run(self, ident, **kw) -> Tuple[torch.Tensor]:
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 0, 0, 1048576)[0]
        delta = parse_param(kw, Lexicon.TIME, EnumConvertType.FLOAT, 0)[0]

        # everybody wang comp tonight
        mode = parse_param(kw, Lexicon.MODE, EnumScaleMode, EnumScaleMode.MATTE.name)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)[0]
        sample = parse_param(kw, Lexicon.SAMPLE, EnumInterpolation, EnumInterpolation.LANCZOS4.name)[0]
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)[0]
        edge_x = parse_param(kw, Lexicon.EDGE_X, EnumGLSLEdge, EnumGLSLEdge.CLAMP.name)[0]
        edge_y = parse_param(kw, Lexicon.EDGE_Y, EnumGLSLEdge, EnumGLSLEdge.CLAMP.name)[0]
        edge = (edge_x, edge_y)

        try:
            self.__glsl.vertex = getattr(self, 'VERTEX', kw.pop(Lexicon.PROG_VERT, None))
            self.__glsl.fragment = getattr(self, 'FRAGMENT', kw.pop(Lexicon.PROG_FRAG, None))
        except CompileException as e:
            comfy_message(ident, "jovi-glsl-error", {"id": ident, "e": str(e)})
            logger.error(self.NAME)
            logger.error(e)
            return

        variables = kw.copy()
        for p in [Lexicon.MODE, Lexicon.WH, Lexicon.SAMPLE, Lexicon.MATTE, Lexicon.BATCH, Lexicon.TIME, Lexicon.FPS, Lexicon.EDGE]:
            variables.pop(p, None)

        self.__glsl.fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1, 120)[0]
        if batch > 0 or self.__delta != delta:
            self.__delta = delta
        step = 1. / self.__glsl.fps

        images = []
        vars = {}
        batch = max(1, batch)
        firstImage = None
        # check if the input(s) have more than a single entry, get the max...
        if batch == 1:
            for k, var in variables.items():
                if isinstance(var, (torch.Tensor)):
                    batch = max(batch, var.shape[0])
                    var = [image_convert(tensor2cv(v), 4) for v in var]
                    if firstImage is None:
                        firstImage = var[0]
                elif isinstance(var, (list, tuple,)):
                    batch = max(batch, len(var))

                variables[k] = var if isinstance(var, (list, tuple,)) else [var]

        pbar = ProgressBar(batch)
        for idx in range(batch):
            for k, val in variables.items():
                vars[k] = val[idx % len(val)]

            w, h = wihi
            if firstImage is not None and mode == EnumScaleMode.MATTE:
                h, w = firstImage.shape[:2]

            self.__glsl.size = (w, h)

            img = self.__glsl.render(self.__delta, edge, **vars)
            if mode != EnumScaleMode.MATTE:
                img = image_scalefit(img, w, h, mode, sample)
            images.append(cv2tensor_full(img, matte))
            self.__delta += step
            comfy_message(ident, "jovi-glsl-time", {"id": ident, "t": self.__delta})
            pbar.update_absolute(idx)
        return [torch.stack(i) for i in zip(*images)]

class GLSLNode(GLSLNodeBase):
    NAME = "GLSL (JOV) 🍩"
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/{JOV_CATEGORY}"
    DESCRIPTION = """
Execute custom GLSL (OpenGL Shading Language) fragment shaders to generate images or apply effects. GLSL is a high-level shading language used for graphics programming, particularly in the context of rendering images or animations. This node allows for real-time rendering of shader effects, providing flexibility and creative control over image processing pipelines. It takes advantage of GPU acceleration for efficient computation, enabling the rapid generation of complex visual effects.
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        opts = d.get('optional', {})
        opts.update({
            Lexicon.BATCH: ("INT", {"default": 0, "mij": 0, "maj": 1048576}),
            Lexicon.FPS: ("INT", {"default": 24, "mij": 1, "maj": 120}),
            Lexicon.TIME: ("FLOAT", {"default": 0, "step": 0.0001, "mij": 0}),
            Lexicon.PROG_VERT: ("STRING", {"default": PROG_VERTEX, "multiline": True, "dynamicPrompts": False}),
            Lexicon.PROG_FRAG: ("STRING", {"default": PROG_FRAGMENT, "multiline": True, "dynamicPrompts": False}),
        })
        d['optional'] = opts
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("NaN")

class GLSLNodeDynamic(GLSLNodeBase):

    PARAM = None

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        original_params = super().INPUT_TYPES()
        opts = original_params.get('optional', {})
        opts.update({
            Lexicon.PROG_FRAG: ("JDATABUCKET", {"fragment": cls.FRAGMENT}),
        })

        # parameter list first...
        data = {}
        if cls.PARAM is not None:
            # 1., 1., 1.; 0; 1; 0.01; rgb | End of the Range
            # default, min, max, step, metadata, tooltip
            for glsl_type, name, default, val_min, val_max, val_step, meta, tooltip in cls.PARAM:
                typ = PTYPE[glsl_type]
                params = {"default": None}

                d = None
                type_name = JOV_TYPE_IMAGE
                if glsl_type != 'sampler2D':
                    type_name = typ.name
                    if default is not None:
                        if default.startswith('EnumGLSL'):
                            if (target_enum := getattr(glsl_enums, default.strip(), None)) is not None:
                                # this be an ENUM....
                                type_name = target_enum._member_names_
                                params['default'] = type_name[0]
                            else:
                                params['default'] = 0
                        else:
                            d = default.split(',')
                            params['default'] = parse_value(d, typ, 0)

                    if val_min is not None:
                        params['mij'] = parse_value(val_min, EnumConvertType.FLOAT, -sys.maxsize)

                    if val_max is not None:
                        params['maj'] = parse_value(val_max, EnumConvertType.FLOAT, sys.maxsize)

                    if val_step is not None:
                        d = 1 if typ.name.endswith('INT') else 0.01
                        params['step'] = parse_value(val_step, EnumConvertType.FLOAT, d)

                    if meta is not None:
                        if "rgb" in meta:
                            if glsl_type.startswith('vec'):
                                params['linear'] = True
                            else:
                                params['rgb'] = True

                if tooltip is not None:
                    params["tooltips"] = tooltip
                data[name] = (type_name, params,)

        data.update(opts)
        original_params['optional'] = data
        return Lexicon._parse(original_params, cls)

def import_dynamic() -> Tuple[str,...]:
    ret = []
    sort = 10000
    root = str(JOV_ROOT_GLSL)
    for name, fname in GLSL_PROGRAMS['fragment'].items():
        if (shader := load_file_glsl(fname)) is None:
            logger.error(f"missing shader file {fname}")
            continue
        meta = shader_meta(shader)
        if meta.get('hide', False):
            continue

        name = meta.get('name', name.split('.')[0]).upper()
        class_name = name.title().replace(' ', '_')
        class_name = f'GLSLNode_{class_name}'

        emoji = Lexicon.GLSL_CUSTOM
        sort_order = sort
        if fname.startswith(root):
            emoji = Lexicon.GLSL_INTERNAL
            sort_order -= 10000

        category = GLSLNodeDynamic.CATEGORY
        if (sub := meta.get('category', None)) is not None:
            category += f'/{sub}'

        class_def = type(class_name, (GLSLNodeDynamic,), {
            "NAME": f'GLSL {name} (JOV) {emoji}'.upper(),
            "DESCRIPTION": meta.get('desc', name),
            "CATEGORY": category.upper(),
            "FRAGMENT": shader,
            "PARAM": meta.get('_', []),
            "SORT": sort_order,
        })

        sort += 10
        ret.append((class_name, class_def,))
    return ret
