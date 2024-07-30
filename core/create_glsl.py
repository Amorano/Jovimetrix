"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Creation
"""

import os
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

from Jovimetrix import JOVImageNode, Lexicon, comfy_message, ROOT
from Jovimetrix.sup.util import load_file, parse_param, EnumConvertType, parse_value
from Jovimetrix.sup.image import EnumInterpolation, EnumScaleMode, cv2tensor_full, image_convert, tensor2cv, MIN_IMAGE_SIZE
from Jovimetrix.sup.shader import PTYPE, shader_meta, CompileException, GLSLShader

# =============================================================================

JOV_ROOT_GLSL = ROOT / 'res' / 'glsl'
GLSL_PROGRAMS = {
    "vertex": {  },
    "fragment": { }
}

GLSL_PROGRAMS['vertex'].update({str(f.relative_to(JOV_ROOT_GLSL)): str(f) for f in Path(JOV_ROOT_GLSL).rglob('*.vert')})
USER_GLSL = ROOT / 'glsl'
USER_GLSL.mkdir(parents=True, exist_ok=True)
if (USER_GLSL := os.getenv("JOV_GLSL", str(USER_GLSL))) is not None:
    GLSL_PROGRAMS['vertex'].update({str(f.relative_to(USER_GLSL)): str(f) for f in Path(USER_GLSL).rglob('*.vert')})

GLSL_PROGRAMS['fragment'].update({str(f.relative_to(JOV_ROOT_GLSL)): str(f) for f in Path(JOV_ROOT_GLSL).rglob('*.frag')})
if USER_GLSL is not None:
    GLSL_PROGRAMS['fragment'].update({str(f.relative_to(USER_GLSL)): str(f) for f in Path(USER_GLSL).rglob('*.frag')})

logger.info(f"  vertex programs: {len(GLSL_PROGRAMS['vertex'])}")
logger.info(f"fragment programs: {len(GLSL_PROGRAMS['fragment'])}")

JOV_CATEGORY = "CREATE"

# =============================================================================

try:
    @PromptServer.instance.routes.get("/jovimetrix/glsl")
    async def jovimetrix_glsl_list(request) -> Any:
        ret = {k:[kk for kk, vv in v.items() \
                  if kk not in ['NONE'] and vv not in [None] and Path(vv).exists()]
               for k, v in GLSL_PROGRAMS.items()}
        return web.json_response(ret)

    @PromptServer.instance.routes.get("/jovimetrix/glsl/{shader}")
    async def jovimetrix_glsl_raw(request, shader:str) -> Any:
        if (program := GLSL_PROGRAMS.get(shader, None)) is None:
            return web.json_response(f"no program {shader}")
        response = load_file(program)
        return web.json_response(response)

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
                response[who] = load_file(data)
            else:
                logger.warning(f"no glsl shader entry {fname}")

        return web.json_response(response)
except Exception as e:
    logger.error(e)

class GLSLNodeBase(JOVImageNode):
    CATEGORY = f"JOVIMETRIX 🔺🟩🔵/GLSL"
    FRAGMENT = None
    VERTEX = None

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = super().INPUT_TYPES()
        d.update({
            "optional": {
                Lexicon.MODE: (EnumScaleMode._member_names_, {"default": EnumScaleMode.NONE.name}),
                Lexicon.WH: ("VEC2", {"default": (512, 512), "min":MIN_IMAGE_SIZE,
                                    "step": 1, "label": [Lexicon.W, Lexicon.H]}),
                Lexicon.SAMPLE: (EnumInterpolation._member_names_, {"default": EnumInterpolation.LANCZOS4.name}),
                Lexicon.MATTE: ("VEC4", {"default": (0, 0, 0, 255), "step": 1,
                                         "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A], "rgb": True})
            }
        })
        return Lexicon._parse(d, cls)

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float("nan")

    def __init__(self, *arg, **kw) -> None:
        super().__init__(*arg, **kw)
        self.__glsl = GLSLShader()
        self.__delta = 0

    def run(self, ident, **kw) -> tuple[torch.Tensor]:
        batch = parse_param(kw, Lexicon.BATCH, EnumConvertType.INT, 0, 0, 1048576)[0]
        delta = parse_param(kw, Lexicon.TIME, EnumConvertType.FLOAT, 0)[0]
        self.__glsl.fps = parse_param(kw, Lexicon.FPS, EnumConvertType.INT, 24, 1, 120)[0]
        wihi = parse_param(kw, Lexicon.WH, EnumConvertType.VEC2INT, [(512, 512)], MIN_IMAGE_SIZE)[0]
        matte = parse_param(kw, Lexicon.MATTE, EnumConvertType.VEC4INT, [(0, 0, 0, 255)], 0, 255)[0]

        variables = kw.copy()
        for p in [Lexicon.WH, Lexicon.MATTE]:
            variables.pop(p, None)

        self.__glsl.size = wihi
        try:
            self.__glsl.program(self.VERTEX, self.FRAGMENT)
        except CompileException as e:
            comfy_message(ident, "jovi-glsl-error", {"id": ident, "e": str(e)})
            logger.error(self.NAME)
            logger.error(e)
            return

        if batch > 0:
            self.__delta = delta
        step = 1. / self.__glsl.fps

        images = []
        pbar = ProgressBar(batch)
        count = batch if batch > 0 else 1
        for idx in range(count):
            vars = {}
            for k, v in variables.items():
                var = v if not isinstance(v, (list, tuple,)) else v[idx] if idx < len(v) else v[-1]
                if isinstance(var, (torch.Tensor)):
                    var = tensor2cv(var)
                    var = image_convert(var, 4)
                vars[k] = var

            image = self.__glsl.render(self.__delta, **vars)
            image = cv2tensor_full(image, matte)
            images.append(image)

            self.__delta += step
            comfy_message(ident, "jovi-glsl-time", {"id": ident, "t": self.__delta})
            pbar.update_absolute(idx)
        return [torch.cat(i, dim=0) for i in zip(*images)]

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
            Lexicon.BATCH: ("INT", {"default": 0, "step": 1, "min": 0, "max": 1048576}),
            Lexicon.FPS: ("INT", {"default": 24, "step": 1, "min": 1, "max": 120}),
            Lexicon.TIME: ("FLOAT", {"default": 0, "step": 0.001, "min": 0, "precision": 4}),
            Lexicon.PROG_VERT: ("STRING", {"default": GLSLShader.PROG_VERTEX, "multiline": True, "dynamicPrompts": False}),
            Lexicon.PROG_FRAG: ("STRING", {"default": GLSLShader.PROG_FRAGMENT, "multiline": True, "dynamicPrompts": False}),
        })
        d['optional'] = opts
        return Lexicon._parse(d, cls)

    def run(self, ident, **kw) -> tuple[torch.Tensor]:
        self.VERTEX = parse_param(kw, Lexicon.PROG_VERT, EnumConvertType.STRING, "")[0]
        self.FRAGMENT = parse_param(kw, Lexicon.PROG_FRAG, EnumConvertType.STRING, "")[0]
        variables = kw.copy()
        for p in [Lexicon.PROG_VERT, Lexicon.PROG_FRAG]:
            variables.pop(p, None)
        return super().run(ident, **variables)

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
            for glsl_type, name, default, tooltip, val_min, val_max, val_step in cls.PARAM:
                typ = PTYPE[glsl_type]
                d = None
                if glsl_type != 'sampler2D' and default is not None:
                    d = default.split(',')
                    d = parse_value(d, typ, 0)

                #val_min = -2147483647
                #val_max = 2147483647
                #val_step = 1
                entry = (typ.name, {})
                match typ:
                    case EnumConvertType.INT | EnumConvertType.VEC2INT | EnumConvertType.VEC3INT | EnumConvertType.VEC4INT:
                        entry = (typ.name, {"default": d, "min": val_min, "max":val_max, "step": val_step, "tooltip": tooltip},)
                    case EnumConvertType.FLOAT | EnumConvertType.VEC2 | EnumConvertType.VEC3 | EnumConvertType.VEC4:
                        entry = (typ.name, {"default": d, "min": val_min, "max":val_max, "step": val_step, "precision": 6, "round": 0.0001, "tooltip": tooltip},)
                    case EnumConvertType.IMAGE:
                        entry = (typ.name, {"default": d, "tooltip": tooltip},)
                data[name] = entry

        data.update(opts)
        original_params['optional'] = data
        return Lexicon._parse(original_params, cls)

def import_dynamic() -> Tuple[str,...]:
    ret = []
    global GLSL_PROGRAMS
    if (prog := GLSL_PROGRAMS['vertex'].pop('_', None)) is not None:
        if (shader := load_file(prog)) is not None:
            GLSLShader.PROG_VERTEX = shader

    if (prog := GLSL_PROGRAMS['fragment'].pop('_', None)) is not None:
        if (shader := load_file(prog)) is not None:
            GLSLShader.PROG_FRAGMENT = shader

    sort = 10000
    root = str(JOV_ROOT_GLSL)
    for name, fname in GLSL_PROGRAMS['fragment'].items():
        if (shader := load_file(fname)) is None:
            logger.error(f"missing shader file {fname}")
            continue

        meta = shader_meta(shader)
        if meta.get('hide', False):
            continue

        name = meta.get('name', name.split('.')[0])
        class_name = name.title().replace(' ', '_')
        class_name = f'GLSLNode_{class_name}'

        emoji = '🧙🏽‍♀️'
        sort_order = sort
        if fname.startswith(root):
            emoji = '🧙🏽'
            sort_order -= 10000

        class_def = type(class_name, (GLSLNodeDynamic,), {
            "NAME": f'GLSL {name} (JOV) {emoji}'.upper(),
            "DESCRIPTION": meta.get('desc', name),
            "FRAGMENT": shader,
            "PARAM": meta.get('_', []),
            "SORT": sort_order,
        })

        sort += 10
        ret.append((class_name, class_def,))
    return ret
