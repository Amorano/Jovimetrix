"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
GLSL Support
"""

import os
import time

import moderngl
import numpy as np
from PIL import Image
from loguru import logger
from sympy import false

# =============================================================================

MAX_WIDTH = 8192
MAX_HEIGHT = 8192

VERTEX = """
#version 330
in vec2 iPosition;
out vec2 fragCoord;

void main() {
    gl_Position = vec4(iPosition, 0.0, 1.0);
    fragCoord = iPosition / 2.0 + 0.5;
}"""

FRAGMENT_HEADER = """
#version 330
#ifdef GL_ES
precision mediump float;
#else
precision highp float;
#endif

in vec2 fragCoord;
uniform vec2 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrameRate;
uniform int iFrame;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;

//#define texture2D texture
layout(location = 0) out vec4 fragColor;
"""

MIN_IMAGE_SIZE = 128

# =============================================================================

class CompileException(Exception): pass

# =============================================================================

class GLSL:
    CTX = None
    VBO = None

    @classmethod
    def instant(cls, fpath: str, texture1:Image=None, width:int=None, height:int=None, param:dict=None) -> Image:
        width = width or MIN_IMAGE_SIZE
        height = height or MIN_IMAGE_SIZE
        if texture1 is not None:
            width, height = texture1.size

        with open(fpath, 'r', encoding='utf8') as f:
            program = f.read()

        # fire and forget
        glsl = GLSL(program, width, height, param=param)
        img = glsl.render(texture1)
        del glsl
        return img

    def __init__(self, fragment:str, width:int=128, height:int=128, param:dict=None) -> None:
        # single initialize for all
        if GLSL.CTX is None:
            GLSL.CTX = moderngl.create_context(standalone=True)
            vertices = np.array([
                -1.0, -1.0,
                1.0, -1.0,
                -1.0,  1.0,
                1.0, -1.0,
                1.0,  1.0,
                -1.0,  1.0
            ], dtype='f4')
            GLSL.VBO = GLSL.CTX.buffer(vertices.tobytes())

        if os.path.isfile(fragment):
            with open(fragment, 'r', encoding='utf8') as f:
                fragment = f.read()
        self.__fragment: str = FRAGMENT_HEADER + fragment
        try:
            self.__prog = GLSL.CTX.program(
                vertex_shader=VERTEX,
                fragment_shader=self.__fragment,
            )
        except Exception as e:
            raise CompileException(e)

        self.__iResolution: tuple[int, int] = self.__prog.get('iResolution', None)
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
