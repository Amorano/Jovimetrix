"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
GLSL Support

Blended from old ModernGL implementation + Audio_Scheduler & Fill Node Pack
"""

import os
import re
import sys
from pathlib import Path
from enum import Enum, EnumType
from typing import Any, Dict, Tuple

import cv2
import glfw
import numpy as np
import OpenGL.GL as gl

from loguru import logger

from Jovimetrix import ROOT
from Jovimetrix.sup.util import EnumConvertType, parse_value, load_file
from Jovimetrix.sup.image import image_convert

# =============================================================================

IMAGE_SIZE_DEFAULT = 512
IMAGE_SIZE_MIN = 64
IMAGE_SIZE_MAX = 16384

LAMBDA_UNIFORM = {
    'bool': gl.glUniform1i,
    'int': gl.glUniform1i,
    'ivec2': gl.glUniform2i,
    'ivec3': gl.glUniform3i,
    'ivec4': gl.glUniform4i,
    'float': gl.glUniform1f,
    'vec2': gl.glUniform2f,
    'vec3': gl.glUniform3f,
    'vec4': gl.glUniform4f,
}

PTYPE = {
    'bool': EnumConvertType.BOOLEAN,
    'int': EnumConvertType.INT,
    'ivec2': EnumConvertType.VEC2INT,
    'ivec3': EnumConvertType.VEC3INT,
    'ivec4': EnumConvertType.VEC4INT,
    'float': EnumConvertType.FLOAT,
    'vec2': EnumConvertType.VEC2,
    'vec3': EnumConvertType.VEC3,
    'vec4': EnumConvertType.VEC4,
    'sampler2D': EnumConvertType.IMAGE
}

class EnumGLSLEdge(Enum):
    CLAMP  = 10
    WRAP   = 20
    MIRROR = 30

class EnumGLSLColorConvert(Enum):
    RGB2HSV = 0
    RGB2LAB = 1
    RGB2XYZ = 2
    HSV2RGB = 10
    HSV2LAB = 11
    HSV2XYZ = 12
    LAB2RGB = 20
    LAB2HSV = 21
    LAB2XYZ = 22
    XYZ2RGB = 30
    XYZ2HSV = 31
    XYZ2LAB = 32

JOV_ROOT_GLSL = ROOT / 'res' / 'glsl'
GLSL_PROGRAMS = {
    "vertex": {  },
    "fragment": { }
}

GLSL_PROGRAMS['vertex'].update({str(f.relative_to(JOV_ROOT_GLSL).as_posix()):
                                str(f) for f in Path(JOV_ROOT_GLSL).rglob('*.vert')})
USER_GLSL = ROOT / 'glsl'
USER_GLSL.mkdir(parents=True, exist_ok=True)
if (USER_GLSL := os.getenv("JOV_GLSL", str(USER_GLSL))) is not None:
    GLSL_PROGRAMS['vertex'].update({str(f.relative_to(USER_GLSL).as_posix()):
                                    str(f) for f in Path(USER_GLSL).rglob('*.vert')})

GLSL_PROGRAMS['fragment'].update({str(f.relative_to(JOV_ROOT_GLSL).as_posix()):
                                  str(f) for f in Path(JOV_ROOT_GLSL).rglob('*.frag')})
if USER_GLSL is not None:
    GLSL_PROGRAMS['fragment'].update({str(f.relative_to(USER_GLSL).as_posix()):
                                      str(f) for f in Path(USER_GLSL).rglob('*.frag')})

try:
    prog = GLSL_PROGRAMS['vertex'].pop('.lib/_.vert')
    PROG_VERTEX = load_file(prog)
except Exception as e:
    logger.error(e)
    raise Exception("failed load default vertex program .lib/_.vert")

try:
    prog = GLSL_PROGRAMS['fragment'].pop('.lib/_.frag')
    PROG_FRAGMENT = load_file(prog)
except Exception as e:
    logger.error(e)
    raise Exception("failed load default fragment program .lib/_.frag")

PROG_HEADER = load_file(JOV_ROOT_GLSL / '.lib/_.head')
PROG_FOOTER = load_file(JOV_ROOT_GLSL / '.lib/_.foot')

logger.info(f"  vertex programs: {len(GLSL_PROGRAMS['vertex'])}")
logger.info(f"fragment programs: {len(GLSL_PROGRAMS['fragment'])}")

RE_INCLUDE = re.compile(r"^\s+?#include\s+?([A-Za-z\_\-\.\\\/]{3,})$", re.MULTILINE)
RE_VARIABLE = re.compile(r"uniform\s+(\w+)\s+(\w+);\s*(?:\/\/\s*([^;|]*))?\s*(?:;\s*([^;|]*))?\s*(?:;\s*([^;|]*))?\s*(?:;\s*([^;|]*))?\s*(?:;\s*([^;|]*))?\s*(?:\|\s*(.*))?$", re.MULTILINE)
RE_SHADER_META = re.compile(r"^\/\/\s?([A-Za-z_]{3,}):\s?(.+)$", re.MULTILINE)

# =============================================================================

class CompileException(Exception): pass

class GLSLShader:

    def __init__(self, vertex:str=None, fragment:str=None, width:int=IMAGE_SIZE_DEFAULT, height:int=IMAGE_SIZE_DEFAULT, fps:int=30) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW did not init")
        self.__size: Tuple[int, int] = (max(width, IMAGE_SIZE_MIN), max(height, IMAGE_SIZE_MIN))
        self.__empty_image: np.ndarray = np.zeros((self.__size[1], self.__size[0]), np.uint8)
        self.__program = None
        self.__source_vertex: str = None
        self.__source_fragment: str = None
        self.__source_vertex_raw: str = None
        self.__source_fragment_raw: str = None
        self.__runtime: float = 0
        self.__fps: int = min(120, max(1, fps))
        self.__mouse: Tuple[int, int] = (0, 0)
        self.__last_frame = np.zeros((self.__size[1], self.__size[0]), np.uint8)
        self.__shaderVar = {}
        self.__userVar = {}
        self.__fbo = None
        self.__fbo_texture = None
        self.__bgcolor = (0, 0, 0, 1.)
        self.__textures = {}
        self.__window = None
        self.__init_window(vertex, fragment)

    def __cleanup(self) -> None:
        glfw.make_context_current(self.__window)
        old = [v[3] for v in self.__userVar.values() if v[0] == 'sampler2D']
        if len(old):
            gl.glDeleteTextures(old)

        if self.__fbo_texture:
            gl.glDeleteTextures(1, [self.__fbo_texture])

        if self.__fbo:
            gl.glDeleteFramebuffers(1, [self.__fbo])

        if self.__program:
            gl.glDeleteProgram(self.__program)

        if self.__window:
            glfw.destroy_window(self.__window)
        logger.debug("cleanup")

    def __init_window(self, vertex:str=None, fragment:str=None, force:bool=False) -> None:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.__cleanup()
        self.__window = glfw.create_window(self.__size[0], self.__size[1], "hidden", None, None)
        if not self.__window:
            raise RuntimeError("GLFW did not init window")

        self.__init_framebuffer()
        self.__init_program(vertex, fragment, force)
        logger.debug("init window")

    def __compile_shader(self, source:str, shader_type:str) -> None:
        glfw.make_context_current(self.__window)
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            log = gl.glGetShaderInfoLog(shader).decode()
            logger.error(f"Shader compilation error: {log}")
            raise CompileException(log)
        # logger.debug(f"{shader_type} compiled")
        return shader

    def __init_program(self, vertex:str=None, fragment:str=None, force:bool=False) -> None:
        vertex = self.__source_vertex_raw if vertex is None else vertex
        if vertex is None:
            logger.debug("Vertex program is empty. Using Default.")
            vertex = PROG_VERTEX

        if (fragment := self.__source_fragment_raw if fragment is None else fragment) is None:
            logger.debug("Fragment program is empty. Using Default.")
            fragment = PROG_FRAGMENT

        if not force and vertex == self.__source_vertex_raw and fragment == self.__source_fragment_raw:
            return

        glfw.make_context_current(self.__window)
        try:
            gl.glDeleteProgram(self.__program)
        except Exception as e:
            pass

        self.__source_vertex = self.__compile_shader(vertex, gl.GL_VERTEX_SHADER)
        fragment_full = PROG_HEADER + fragment + PROG_FOOTER
        self.__source_fragment = self.__compile_shader(fragment_full, gl.GL_FRAGMENT_SHADER)

        self.__program = gl.glCreateProgram()
        gl.glAttachShader(self.__program, self.__source_vertex)
        gl.glAttachShader(self.__program, self.__source_fragment)
        gl.glLinkProgram(self.__program)
        if gl.glGetProgramiv(self.__program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            log = gl.glGetProgramInfoLog(self.__program).decode()
            logger.error(f"Program linking error: {log}")
            raise RuntimeError(log)

        self.__source_fragment_raw = fragment
        self.__source_vertex_raw = vertex

        gl.glUseProgram(self.__program)

        self.__shaderVar = {}
        statics = ['iResolution', 'iTime', 'iFrameRate', 'iFrame']
        for s in statics:
            if (val := gl.glGetUniformLocation(self.__program, s)) > -1:
                self.__shaderVar[s] = val

        if (resolution := self.__shaderVar.get('iResolution', -1)) > -1:
            gl.glUniform3f(resolution, self.__size[0], self.__size[1], 0)

        self.__userVar = {}
        # read the fragment and setup the vars....
        for match in RE_VARIABLE.finditer(self.__source_fragment_raw):
            typ, name, default, val_min, val_max, val_step, meta, tooltip = match.groups()

            texture = None
            if typ in ['sampler2D']:
                texture = self.__textures[name] = gl.glGenTextures(1)
            else:
                default = default.strip()
                if default.startswith('EnumGLSL'):
                    typ = 'int'
                    if (target_enum := getattr(sys.modules[__name__], default, None)) is not None:
                        default = target_enum
                    else:
                        default = 0

            # logger.debug(f"{name}.{typ}: {default} {val_min} {val_max} {val_step} {tooltip}")
            self.__userVar[name] = [
                # type
                typ,
                # gl location
                gl.glGetUniformLocation(self.__program, name),
                # default value
                default,
                # texture id -- if a texture
                texture
            ]

        logger.debug("init vars")
        logger.debug("init program")

    def __init_framebuffer(self) -> None:
        glfw.make_context_current(self.__window)

        self.__fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)

        self.__fbo_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.__fbo_texture)
        glfw.set_window_size(self.__window, self.__size[0], self.__size[1])

        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, self.__size[0], self.__size[1], 0, gl.GL_RGBA, gl.GL_FLOAT, None)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.__fbo_texture, 0)

        gl.glViewport(0, 0, self.__size[0], self.__size[1])

        self.__empty_image = np.zeros((self.__size[1], self.__size[0]), np.uint8)
        logger.debug("init framebuffer")

    def __del__(self) -> None:
        self.__cleanup()
        #if self.__window is not None:
        #    if glfw is not None:
        #        glfw.destroy_window(self.__window)
        #    self.__window = None
        # glfw.terminate()

    @property
    def vertex(self) -> str:
        return self.__source_vertex_raw

    @vertex.setter
    def vertex(self, program:str) -> None:
        self.__init_program(vertex=program)

    @property
    def fragment(self) -> str:
        return self.__source_fragment_raw

    @fragment.setter
    def fragment(self, program:str) -> None:
        self.__init_program(fragment=program)

    @property
    def size(self) -> Tuple[int, int]:
        return self.__size

    @size.setter
    def size(self, size:Tuple[int, int]) -> None:
        size = (min(IMAGE_SIZE_MAX, max(IMAGE_SIZE_MIN, size[0])),
                min(IMAGE_SIZE_MAX, max(IMAGE_SIZE_MIN, size[1])))

        if size[0] != self.__size[0] or size[1] != self.__size[1]:
            self.__size = size
            self.__init_window(force=True)

    @property
    def runtime(self) -> float:
        return self.__runtime

    @runtime.setter
    def runtime(self, runtime:float) -> None:
        runtime = max(0, runtime)
        self.__runtime = runtime

    @property
    def fps(self) -> int:
        return self.__fps

    @fps.setter
    def fps(self, fps:int) -> None:
        fps = max(1, min(120, int(fps)))
        self.__fps = fps
        if (iFrameRate := self.__shaderVar.get('iFrameRate', -1)) > -1:
            glfw.make_context_current(self.__window)
            gl.glUseProgram(self.__program)
            gl.glUniform1f(self.__shaderVar['iFrameRate'], iFrameRate)

    @property
    def mouse(self) -> Tuple[int, int]:
        return self.__mouse

    @mouse.setter
    def mouse(self, pos:Tuple[int, int]) -> None:
        self.__mouse = pos

    @property
    def frame(self) -> float:
        return int(self.__runtime * self.__fps)

    @property
    def last_frame(self) -> float:
        return self.__last_frame

    @property
    def bgcolor(self) -> Tuple[int, ...]:
        return self.__bgcolor

    @bgcolor.setter
    def bgcolor(self, color:Tuple[int, ...]) -> None:
        self.__bgcolor = tuple(float(x) / 255. for x in color)

    def render(self, time_delta:float=0.,
               tile_edge:Tuple[EnumGLSLEdge,...]=(EnumGLSLEdge.CLAMP, EnumGLSLEdge.CLAMP),
               **kw) -> np.ndarray:

        glfw.make_context_current(self.__window)
        gl.glUseProgram(self.__program)

        self.runtime = time_delta

        # current time in shader lifetime
        if (val := self.__shaderVar.get('iTime', -1)) > -1:
            gl.glUniform1f(val, self.__runtime)

        # the desired FPS
        if (val := self.__shaderVar.get('iFrameRate', -1)) > -1:
            gl.glUniform1i(val, self.__fps)

        # the current frame based on the life time and "fps"
        if (val := self.__shaderVar.get('iFrame', -1)) > -1:
            gl.glUniform1i(val, self.frame)

        texture_index = 0
        for uk, uv in self.__userVar.items():
            p_type, p_loc, p_value, _ = uv
            val = kw.get(uk, p_value)

            if p_type == 'sampler2D':
                if (texture := self.__textures.get(uk, None)) is None:
                    logger.error(f"texture [{texture_index}] {uk} is None")
                    texture_index += 1
                    continue

                gl.glActiveTexture(gl.GL_TEXTURE0 + texture_index)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

                # send in black if nothing in input image
                if not isinstance(val, (np.ndarray,)):
                    val = self.__empty_image

                # @TODO: could cache this ?
                val = image_convert(val, 4)
                val = val[::-1,:]
                val = val.astype(np.float32) / 255.0
                val = cv2.resize(val, self.__size, interpolation=cv2.INTER_LINEAR)
                #
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, self.__size[0], self.__size[1], 0, gl.GL_RGBA, gl.GL_FLOAT, val)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

                for idx, text_wrap in enumerate([gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T]):
                    match EnumGLSLEdge[tile_edge[idx]]:
                        case EnumGLSLEdge.WRAP:
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, text_wrap, gl.GL_REPEAT)
                        case EnumGLSLEdge.MIRROR:
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, text_wrap, gl.GL_MIRRORED_REPEAT)
                        case _:
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, text_wrap, gl.GL_CLAMP_TO_EDGE)

                gl.glUniform1i(p_loc, texture_index)
                texture_index += 1
            elif val:
                funct = LAMBDA_UNIFORM[p_type]
                if isinstance(p_value, EnumType):
                    val = p_value[val].value
                elif isinstance(val, str):
                    val = val.split(',')
                val = parse_value(val, PTYPE[p_type], 0)
                if not isinstance(val, (list, tuple)):
                    val = [val]
                funct(p_loc, *val)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)
        gl.glClearColor(*self.__bgcolor)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

        data = gl.glReadPixels(0, 0, self.__size[0], self.__size[1], gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.__size[1], self.__size[0], 4)
        self.__last_frame = image[::-1, :, :]

        glfw.poll_events()

        return self.__last_frame

def shader_meta(shader: str) -> Dict[str, Any]:
    ret = {}
    for match in RE_SHADER_META.finditer(shader):
        key, value = match.groups()
        ret[key] = value
    ret['_'] = [match.groups() for match in RE_VARIABLE.finditer(shader)]
    return ret

def load_file_glsl(fname: str) -> str:

    # first file we load, starts the list of included
    include = set()

    def scan_include(file:str, idx:int=0) -> str:
        if idx > 4:
            return "too many recursive includes"

        file_path = JOV_ROOT_GLSL / file
        if file_path in include:
            return ""

        include.add(file_path)
        try:
            result = load_file(file_path)
        except FileNotFoundError:
            return f"File not found: {file_path}"

        # replace #include directives with their content
        def replace_include(match):
            lib_path = JOV_ROOT_GLSL / match.group(1)
            if lib_path not in include:
                return scan_include(lib_path, idx+1)
            return ""

        return RE_INCLUDE.sub(replace_include, result)

    return scan_include(fname)
