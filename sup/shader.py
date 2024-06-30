"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
GLSL Support

Blended from old ModernGL implementation + Audio_Scheduler & Fill Node Pack
"""

import time
from typing import Tuple

import glfw
import numpy as np
import OpenGL.GL as gl

from loguru import logger

# =============================================================================

MIN_IMAGE_SIZE = 32
MAX_WIDTH = 8192
MAX_HEIGHT = 8192

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

    uniform sampler2D   iChannel0;
    uniform sampler2D   iChannel1;
    uniform sampler2D   iChannel2;
    uniform sampler2D   iChannel3;

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
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        // Normalized pixel coordinates (from 0 to 1)
        vec2 uv = fragCoord/iResolution.xy;

        // Time varying pixel color
        vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

        // Output to screen
        fragColor = vec4(col,1.0);
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

    def __init__(self, width:int=MIN_IMAGE_SIZE, height:int=MIN_IMAGE_SIZE) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW did not init")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden
        window = glfw.create_window(width, height, "hidden", None, None)
        if not window:
            raise RuntimeError("GLFW did not init window")
        glfw.make_context_current(window)
        #
        self.__fbo = None
        self.__program = None
        self.__source_vertex = None
        self.__source_vertex_raw: str = self.PROG_VERTEX
        self.__source_fragment = None
        self.__source_fragment_raw: str = self.PROG_HEADER + self.PROG_FRAGMENT + self.PROG_FOOTER
        #
        self.__size: Tuple[int, int] = (width, height)
        self.__textures = None
        self.__runtime: float = 0
        self.__delta: float = 0
        self.__fps: int = 30
        self.__frame: int = 0
        self.__frame_count: int = 0
        self.__time_last: float = time.perf_counter()
        self.__compile()

    def __compile_shader(self, source:str, shader_type:str) -> None:
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            raise RuntimeError(gl.glGetShaderInfoLog(shader))
        logger.debug(f"{shader_type} compiled")
        return shader

    def __compile(self) -> None:
        self.__program = gl.glCreateProgram()
        gl.glAttachShader(self.__program, self.__source_vertex)
        gl.glAttachShader(self.__program, self.__source_fragment)
        gl.glLinkProgram(self.__program)
        if gl.glGetProgramiv(self.__program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            raise RuntimeError(gl.glGetProgramInfoLog(self.__program))

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

        #
        self.__textures = gl.glGenTextures(4)
        gl.glUseProgram(self.__program)

    def __del__(self) -> None:
        # assume all other resources get cleaned up with the context
        glfw.terminate()

    @property
    def vertex(self) -> str:
        return self.__source_vertex

    @vertex.setter
    def vertex(self, vertex_source:str) -> None:
        if vertex_source != self.__source_vertex_raw:
            self.__source_vertex = self.__compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
            self.__source_vertex_raw = vertex_source
            self.__compile()

    @property
    def fragment(self) -> str:
        return self.__source_fragment

    @fragment.setter
    def fragment(self, fragment_source:str) -> None:
        fragment_source = self.PROG_HEADER + fragment_source + self.PROG_FOOTER
        if fragment_source != self.__source_fragment_raw:
            self.__source_fragment = self.__compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
            self.__source_fragment_raw = fragment_source
            self.__compile()

    @property
    def size(self) -> Tuple[int, int]:
        return self.__size

    @size.setter
    def size(self, size:Tuple[int, int]) -> None:
        print(self.__program)
        iResolution_location = gl.glGetUniformLocation(self.__program, "iResolution")
        gl.glUniform3f(iResolution_location, size[0], size[1], 0)
        self.__size = size

    @property
    def runtime(self) -> float:
        return self.__runtime

    @runtime.setter
    def runtime(self, runtime:float) -> None:
        iTime_location = gl.glGetUniformLocation(self.__program, "iTime")
        gl.glUniform1f(iTime_location, runtime)
        self.__runtime = runtime

        frame = runtime * self.__fps
        iFrame_location = gl.glGetUniformLocation(self.__program, "iFrame")
        gl.glUniform1i(iFrame_location, frame)
        self.__frame = frame

    @property
    def fps(self) -> float:
        return self.__fps

    @fps.setter
    def fps(self, fps:float) -> None:
        fps = max(1, min(120, fps))
        iFrameRate_location = gl.glGetUniformLocation(self.__program, "iFrameRate")
        gl.glUniform1f(iFrameRate_location, fps)
        self.__fps = fps
        self.__delta = 1.0 / fps

    @property
    def mouse(self) -> Tuple[int, int]:
        return self.__mouse

    @mouse.setter
    def mouse(self, pos:Tuple[int, int]) -> None:
        iMouse_location = gl.glGetUniformLocation(self.__program, "iMouse")
        gl.glUniform4f(iMouse_location, pos[0], pos[1], 0, 0)
        self.__mouse = pos

    @property
    def delta(self) -> float:
        return self.__delta

    @property
    def frame(self) -> float:
        return self.__frame

    def update_texture(self, texture, image) -> None:
        image = image[::-1,:,:]
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.shape[1], image.shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, image)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    def render(self, time_delta:float=0.) -> np.ndarray:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self.__program)

        # bind textures...
        #for i in range(4):
        #    gl.glActiveTexture(gl.GL_TEXTURE0 + i)  # type: ignore
        #    gl.glBindTexture(gl.GL_TEXTURE_2D, self.__textures[i])
        #    iChannel_location = gl.glGetUniformLocation(self.__program, f"iVar{i}")
        #    gl.glUniform1i(iChannel_location, i)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        data = gl.glReadPixels(0, 0, self.__size[0], self.__size[1], gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.__size[1], self.__size[0], 4)
        image = image[0:2:-1, :, :]
        image = np.array(image).astype(np.float32) / 255.0

        '''
        pixels = self.__fbo.color_attachments[0].read()
        self.__frame = Image.frombytes(
            "RGBA", self.__fbo.size, pixels,
            "raw", "RGBA", 0, -1
        )
        self.__frame = self.__frame.transpose(Image.FLIP_TOP_BOTTOM)
        '''

        # step frame
        if not self.__hold:
            self.__frame_count += 1
            self.__delta = max(0, self.__fps) if self.__fps > 0 else time.perf_counter() - self.__time_last
            self.__runtime += self.__delta
            self.__time_last = time.perf_counter()
        return image

# =============================================================================

'''
class GLSL:
    CTX = None
    VBO = None

    def __init__(self, fragment:str, width:int=128, height:int=128, param:dict=None) -> None:
        # single initialize for all
        if GLSL.CTX is None:
            GLSL.CTX = moderngl.create_context(standalone=True)
            vertices = np.array([
                -1.0, -1.0,
                1.0, -1.0,
                -1.0, 1.0,
                1.0, -1.0,
                1.0, 1.0,
                -1.0, 1.0
            ], dtype='f4')
            GLSL.VBO = GLSL.CTX.buffer(vertices.tobytes())

        if os.path.isfile(fragment):
            with open(fragment, 'r', encoding='utf8') as f:
                fragment = f.read()
        self.__fragment: str =
        + fragment
        try:
            self.__prog = GLSL.CTX.program(
                vertex_shader=VERTEX,
                fragment_shader=self.__fragment,
            )
        except Exception as e:
            raise CompileException(e)

        self.__iResolution: Tuple[int, int] = self.__prog.get('iResolution', None)
        self.__iTime: float = self.__prog.get('iTime', None)
        self.__iTimeDelta: float = self.__prog.get('iTimeDelta', None)
        self.__iFrameRate: float = self.__prog.get('iFrameRate', None)
        self.__iFrame: int = self.__prog.get('iFrame', None)

        try:
            self.__prog['iChannel0'].value = 0
        except:
            pass
        self.__iChannel0 = self.__prog.get('iChannel0', None)

        for k, v in (param or {}).items():
            var = self.__prog.get(k, None)
            if var is None:
                logger.warning(f"variable missing {k}")
                continue

            if isinstance(v, dict):
                v = [v[str(k)] for k in range(len(v))]
            try:
                self.__prog[k].value = v
            except Exception as e:
                logger.error(e)
                logger.warning(k)
                logger.warning(v)

        self.__vao = GLSL.CTX.simple_vertex_array(self.__prog, GLSL.VBO, "iPosition")
        self.__width = width
        self.__height = height
        self.__texture = GLSL.CTX.texture((width, height), 4)
        self.__fbo = GLSL.CTX.framebuffer(
            color_attachments=[self.__texture]
        )

        # FPS > 0 will act as a step (per frame step)
        self.__fps: float = 0
        self.__fps_rate: float = 0
        # the last frame rendered
        self.__frame: Image = Image.new("RGB", (1, 1))
        self.__hold: bool = False
        self.__runtime: float = 0
        self.__delta: float = 0
        self.__frame_count: int = 0
        self.__time_last: float = time.perf_counter()

    def reset(self) -> None:
        self.__runtime = 0
        self.__delta = 0
        self.__frame_count = 0
        self.__time_last = time.perf_counter()

    def __bufferReset(self) -> None:
        if self.__fbo is not None:
            self.__fbo.release()
            self.__fbo = None

        if self.__texture is not None:
            self.__texture.release()
            self.__texture = None

        if self.__texture is None:
            self.__texture = GLSL.CTX.texture((self.__width, self.__height), 4)

        try:
            self.__fbo = GLSL.CTX.framebuffer(
                color_attachments=[self.__texture]
            )
        except Exception as e:
            logger.error(str(e))
            logger.debug(GLSL.CTX)
            logger.debug(self.__width)
            logger.debug(self.__height)

    @property
    def frame(self) -> Image:
        """the current frame."""
        return self.__frame

    @property
    def fps(self) -> int:
        return self.__fps

    @fps.setter
    def fps(self, val:int) -> None:
        self.__fps = max(0, min(1000, val))
        if self.__fps > 0:
            self.__fps_rate = 1 / self.__fps

    @property
    def runtime(self) -> float:
        return self.__runtime

    @runtime.setter
    def runtime(self, val:float) -> None:
        self.__runtime = max(0, val)

    @property
    def hold(self) -> bool:
        return self.__hold

    @hold.setter
    def hold(self, val: bool) -> None:
        self.__hold = val

    @property
    def width(self) -> int:
        return self.__width

    @width.setter
    def width(self, val: int) -> None:
        val = max(0, min(val, MAX_WIDTH))
        if val != self.__width:
            self.__width = val
            self.__bufferReset()

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, val: int) -> None:
        val = max(0, min(val, MAX_HEIGHT))
        if val != self.__height:
            self.__height = val
            self.__bufferReset()

    @property
    def channel0(self) -> int:
        return self.__iChannel0

    @channel0.setter
    def channel0(self, val:np.ndarray) -> None:
        if self.__iChannel0 is not None:
            if len(val.mode) != 4:
                val = val.convert("RGBA")
            self.__channel0_texture = GLSL.CTX.texture(val.size, components=4, data=val.tobytes())
            self.__channel0_texture.use(location=0)

    def __set_uniforms(self) -> None:
        if self.__iResolution is not None:
            self.__iResolution.value = (self.__width, self.__height)

        if self.__iTime is not None:
            self.__iTime.value = self.__runtime

        if self.__iTimeDelta is not None:
            self.__iTimeDelta.value = self.__delta

        if self.__iFrameRate is not None:
            self.__iFrameRate.value = self.__fps_rate

        if self.__iFrame is not None:
            self.__iFrame.value = self.__frame_count

    def render(self, channel0:Image=None, param:dict=None) -> Image:
        self.__fbo.use()
        self.__fbo.clear(0.0, 0.0, 0.0)
        if not self.__hold:
            self.__set_uniforms()
            if channel0 is not None:
                self.channel0 = channel0
            for k, v in (param or {}).items():
                try:
                    self.__prog[k].value = v
                except KeyError as _:
                    pass
                except Exception as e:
                    logger.error(k, v)
                    logger.error(str(e))

        self.__vao.render()
        pixels = self.__fbo.color_attachments[0].read()
        self.__frame = Image.frombytes(
            "RGBA", self.__fbo.size, pixels,
            "raw", "RGBA", 0, -1
        )
        self.__frame = self.__frame.transpose(Image.FLIP_TOP_BOTTOM)

        # step frame
        if not self.__hold:
            self.__frame_count += 1
            self.__delta = max(0, self.__fps_rate) if self.__fps > 0 else time.perf_counter() - self.__time_last
            self.__runtime += self.__delta
            self.__time_last = time.perf_counter()

        return self.__frame
'''

# =============================================================================

