"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
GLSL Support

Blended from old ModernGL implementation + Audio_Scheduler & Fill Node Pack
"""

import time
from typing import Tuple

import cv2
import glfw
import numpy as np
import OpenGL.GL as gl

from loguru import logger

# =============================================================================

IMAGE_SIZE_MIN = 512
IMAGE_SIZE_MAX = 8192

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
        self.__textures = None
        self.__program = None
        self.__source_vertex: None
        self.__source_fragment: None
        self.__source_vertex_raw: str = None
        self.__source_fragment_raw: str = None
        self.__runtime: float = 0
        self.__delta: float = 0
        self.__frame: int = 0
        self.__fps: int = 30
        self.__mouse: Tuple[int, int] = (0, 0)
        self.__last_frame = np.zeros((self.__size[1], self.__size[0]), np.uint8)
        self.__shaderVar = {}
        self.program(vertex, fragment)
        #self.fps = fps
        #self.size = (width, height)
        self.__time_last: float = time.perf_counter()

    def __updateVars(self) -> None:
        gl.glUseProgram(self.__program)
        gl.glUniform3f(self.__shaderVar['iResolution'], self.__size[0], self.__size[1], 0)
        gl.glUniform1f(self.__shaderVar['iTime'], self.__runtime)
        gl.glUniform1f(self.__shaderVar['iFrameRate'], self.__fps)
        gl.glUniform1i(self.__shaderVar['iFrame'], self.frame)
        gl.glUniform4f(self.__shaderVar['iMouse'], self.__mouse[0], self.__mouse[1], 0, 0)

        # print(self.__size, self.__runtime, self.__fps, self.frame, self.__mouse)

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
        # gl.glUseProgram(self.__program)

    def __del__(self) -> None:
        # assume all other resources get cleaned up with the context
        glfw.terminate()

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
        vertex = self.__source_vertex_raw if vertex is None else vertex
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

            self.__textures = gl.glGenTextures(4)
            self.__source_fragment_raw = fragment
            self.__source_vertex_raw = vertex
            self.__shaderVar = {
                'iResolution': gl.glGetUniformLocation(self.__program, "iResolution"),
                'iTime': gl.glGetUniformLocation(self.__program, "iTime"),
                'iFrameRate': gl.glGetUniformLocation(self.__program, "iFrameRate"),
                'iFrame': gl.glGetUniformLocation(self.__program, "iFrame"),
                'iMouse': gl.glGetUniformLocation(self.__program, "iMouse"),
            }

    def update_texture(self, texture, image) -> None:
        image = image[::-1,:,:]
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.shape[1], image.shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, image)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    def render(self, time_delta:float=0.) -> np.ndarray:
        self.runtime = time_delta
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__fbo)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # gl.glUseProgram(self.__program)
        self.__updateVars()

        # bind textures...
        #for i in range(4):
        #    gl.glActiveTexture(gl.GL_TEXTURE0 + i)  # type: ignore
        #    gl.glBindTexture(gl.GL_TEXTURE_2D, self.__textures[i])
        #    iChannel_location = gl.glGetUniformLocation(self.__program, f"iVar{i}")
        #    gl.glUniform1i(iChannel_location, i)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        data = gl.glReadPixels(0, 0, self.__size[0], self.__size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.__size[1], self.__size[0], 3)
        self.__last_frame = image[::-1, :, :]
        return self.__last_frame

# =============================================================================

if __name__ == "__main__":
    g = GLSLShader(width=512, height=512)
    fps = 60.
    delta = 1. / fps
    frame = 1110
    image = np.zeros((512, 512), dtype=np.uint8)
    while 1:
        t = time.perf_counter()
        image = g.render(frame * delta)
        cv2.imshow(f"a", image)
        frame += 1
        if (d := (time.perf_counter() - t)) < delta:
            time.sleep(delta - d)
        glfw.poll_events()
