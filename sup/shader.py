"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
GLSL Support

Blended from old ModernGL implementation + Audio_Scheduler & Fill Node Pack
"""

import re
from typing import Tuple

import glfw
import numpy as np
import OpenGL.GL as gl

from loguru import logger
from Jovimetrix.sup.util import EnumConvertType, parse_value

# =============================================================================

IMAGE_SIZE_MIN = 512
IMAGE_SIZE_MAX = 8192

LAMBDA_UNIFORM = {
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
    'int': EnumConvertType.INT,
    'ivec2': EnumConvertType.VEC2INT,
    'ivec3': EnumConvertType.VEC3INT,
    'ivec4': EnumConvertType.VEC4INT,
    'float': EnumConvertType.FLOAT,
    'vec2': EnumConvertType.VEC2,
    'vec3': EnumConvertType.VEC3,
    'vec4': EnumConvertType.VEC4
}

RE_VARIABLE = re.compile(r"uniform\s*(\w*)\s*(\w*);(?:.*\/{2}\s*([A-Za-z0-9\-\.,\s]+)){0,1}\s*$", re.MULTILINE)

# =============================================================================

class CompileException(Exception): pass

class GLSLShader():
    PROG_HEADER = """
    #version 440

    precision highp float;

    uniform vec3	iResolution;
    uniform vec4	iMouse;
    uniform float	iTime;
    uniform float	iTimeDelta;
    uniform float	iFrameRate;
    uniform int	    iFrame;

    #define texture2D texture
    """

    PROG_FOOTER = """
    layout(location = 0) out vec4 _fragColor;

    void main()
    {
        mainImage(_fragColor, gl_FragCoord.xy);
    }
    """

    PROG_FRAGMENT = """
uniform sampler2D imageA;
uniform sampler2D imageB;
uniform float size; // 7.
uniform float time_scale; // 19.

void mainImage( out vec4 fragColor, vec2 fragCoord ) {
  vec2 u = floor(fragCoord / size);
  float s = dot(u, u) + iTime / time_scale;
  u = mod(u, 2.);

  vec2 uv = fragCoord.xy / iResolution.xy;
  vec3 col2 = texture2D(imageB, uv).rgb;
  vec3 col = texture2D(imageA, uv).rgb;
  fragColor = vec4(col + (floor(sin(s)) - u.y - floor(cos(s)) - u.xxxx).rgb, 1.0);
}
"""

    PROG_VERTEX = """
#version 330 core
void main()
{
    vec2 verts[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(verts[gl_VertexID], 0, 1);
}
"""

    def __init__(self, vertex:str=None, fragment:str=None, width:int=IMAGE_SIZE_MIN, height:int=IMAGE_SIZE_MIN, fps:int=30) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW did not init")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden
        self.__window = glfw.create_window(width, height, "hidden", None, None)
        if not self.__window:
            raise RuntimeError("GLFW did not init window")
        glfw.make_context_current(self.__window)
        #
        self.__size: Tuple[int, int] = (max(width, IMAGE_SIZE_MIN), max(height, IMAGE_SIZE_MIN))
        self.__fbo = None
        self.__program = None
        self.__source_vertex: None
        self.__source_fragment: None
        self.__source_vertex_raw: str = None
        self.__source_fragment_raw: str = None
        self.__runtime: float = 0
        self.__fps: int = 30
        self.__mouse: Tuple[int, int] = (0, 0)
        self.__last_frame = np.zeros((self.__size[1], self.__size[0]), np.uint8)
        self.__shaderVar = {}
        self.__userVar = {}
        self.program(vertex, fragment)

    def __compile_shader(self, source:str, shader_type:str) -> None:
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            raise CompileException(gl.glGetShaderInfoLog(shader))
        logger.debug(f"{shader_type} compiled")
        return shader

    def __framebuffer(self) -> None:
        # match the window to the buffer size...
        glfw.set_window_size(self.__window, self.__size[0], self.__size[1])
        # MAKE FRAMEBUFFER
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.__size[0], self.__size[1], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.__fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer is not complete")

    def __del__(self) -> None:
        # assume all other resources get cleaned up with the context
        """
        try:
            if len(self.__textures):
                gl.glDeleteTextures(len(self.__textures), self.__textures)
        except:
            pass
        """

    @property
    def vertex(self) -> str:
        return self.__source_vertex_raw

    @vertex.setter
    def vertex(self, program:str) -> None:
        if program != self.__source_vertex_raw:
            self.program(vertex=program)

    @property
    def fragment(self) -> str:
        return self.__source_fragment_raw

    @fragment.setter
    def fragment(self, program:str) -> None:
        if program != self.__source_fragment_raw:
            self.program(fragment=program)

    @property
    def size(self) -> Tuple[int, int]:
        return self.__size

    @size.setter
    def size(self, size:Tuple[int, int]) -> None:
        self.__size = (min(IMAGE_SIZE_MAX, max(IMAGE_SIZE_MIN, size[0])),
                min(IMAGE_SIZE_MAX, max(IMAGE_SIZE_MIN, size[1])))
        self.__framebuffer()

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

    def program(self, vertex:str=None, fragment:str=None) -> None:
        if (vertex := self.__source_vertex_raw if vertex is None else vertex) is None:
            logger.debug("Vertex program is empty. Using Default.")
            vertex = self.PROG_VERTEX

        if (fragment := self.__source_fragment_raw if fragment is None else fragment) is None:
            logger.debug("Fragment program is empty. Using Default.")
            fragment = self.PROG_FRAGMENT

        if vertex != self.__source_vertex_raw or fragment != self.__source_fragment_raw:
            self.__source_vertex = self.__compile_shader(vertex, gl.GL_VERTEX_SHADER)
            fragment_full = self.PROG_HEADER + fragment + self.PROG_FOOTER
            self.__source_fragment = self.__compile_shader(fragment_full, gl.GL_FRAGMENT_SHADER)

            self.__program = gl.glCreateProgram()
            gl.glAttachShader(self.__program, self.__source_vertex)
            gl.glAttachShader(self.__program, self.__source_fragment)
            gl.glLinkProgram(self.__program)
            if gl.glGetProgramiv(self.__program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
                raise RuntimeError(gl.glGetProgramInfoLog(self.__program))

            self.__framebuffer()

            self.__source_fragment_raw = fragment
            self.__source_vertex_raw = vertex
            self.__shaderVar = {
                'iResolution': gl.glGetUniformLocation(self.__program, "iResolution"),
                'iTime': gl.glGetUniformLocation(self.__program, "iTime"),
                'iFrameRate': gl.glGetUniformLocation(self.__program, "iFrameRate"),
                'iFrame': gl.glGetUniformLocation(self.__program, "iFrame"),
                'iMouse': gl.glGetUniformLocation(self.__program, "iMouse")
            }

            texture_count = 0
            self.__userVar = {}
            # read the fragment and setup the vars....
            for match in RE_VARIABLE.finditer(fragment):
                typ, name, default = match.groups()
                tex_loc = None
                if typ in ['sampler2D']:
                    tex_loc = gl.glGenTextures(1)
                logger.debug(f"{name}.{typ}: {default}")
                self.__userVar[name] = [
                    # type
                    typ,
                    # gl location
                    gl.glGetUniformLocation(self.__program, name),
                    # default value
                    default,
                    # texture id -- if a texture
                    tex_loc
                ]

    def render(self, time_delta:float=0., **kw) -> np.ndarray:
        self.runtime = time_delta
        gl.glUseProgram(self.__program)

        # SET SHADER STATIC VARS
        gl.glUniform3f(self.__shaderVar['iResolution'], self.__size[0], self.__size[1], 0)
        gl.glUniform1f(self.__shaderVar['iTime'], self.__runtime)
        gl.glUniform1f(self.__shaderVar['iFrameRate'], self.__fps)
        gl.glUniform1i(self.__shaderVar['iFrame'], self.frame)
        gl.glUniform4f(self.__shaderVar['iMouse'], self.__mouse[0], self.__mouse[1], 0, 0)

        empty = np.zeros((self.__size[0], self.__size[1], 4), dtype=np.uint8)

        # SET USER DYNAMIC VARS
        # update any user vars...
        texture_index = 0
        for uk, uv in self.__userVar.items():
            # type, loc, value, index
            p_type, p_loc, p_value, p_tex = uv
            # use the default....
            val = p_value if not uk in kw else kw[uk]

            # SET TEXTURE
            if (p_type == 'sampler2D'):
                # cache textures? or do we care per frame?
                gl.glBindTexture(gl.GL_TEXTURE_2D, p_tex)
                val = empty if val is None else val
                if val.ndim == 3:
                    op = gl.GL_RGBA if val.shape[2] == 4 else gl.GL_RGB
                    val = val[::-1,:]
                    val = val.astype(np.float32) / 255.0
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, op, val.shape[1], val.shape[0], 0, op, gl.GL_FLOAT, val)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                # Bind the texture to the texture unit
                gl.glActiveTexture(gl.GL_TEXTURE0 + texture_index)
                gl.glBindTexture(gl.GL_TEXTURE_2D, p_tex)
                gl.glUniform1i(p_loc, texture_index)
                texture_index += 1
            else:
                funct = LAMBDA_UNIFORM.get(p_type)
                if funct is not None:
                    val = parse_value(val, PTYPE[p_type], 0)
                    if not isinstance(val, (list, tuple)):
                        val = [val]
                    funct(p_loc, *val)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        data = gl.glReadPixels(0, 0, self.__size[0], self.__size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.__size[1], self.__size[0], 3)
        self.__last_frame = image[::-1, :, :]
        return self.__last_frame
