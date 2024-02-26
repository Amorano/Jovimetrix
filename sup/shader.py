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
        self.__fbo = None
        self.__vao = None
        self.__vbo = None
        self.__ctx = moderngl.create_context(standalone=True)

        if os.path.isfile(fragment):
            with open(fragment, 'r', encoding='utf8') as f:
                fragment = f.read()
        self.__fragment: str = FRAGMENT_HEADER + fragment

        try:
            self.__prog = self.__ctx.program(
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

        self.__param = {}
        for k, v in (param or {}).items():
            self.__param[k] = self.__prog.get(k, None)
            if self.__param[k] is not None:
                if isinstance(v, dict):
                    v = [v[str(k)] for k in range(len(v))]
                self.__param[k].value = v

        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype='f4')
        self.__vbo = self.__ctx.buffer(vertices.tobytes())
        self.__vao = self.__ctx.simple_vertex_array(self.__prog, self.__vbo, "iPosition")

        self.__width = width
        self.__height = height
        self.__texture = self.__ctx.texture((width, height), 3)
        self.__fbo = self.__ctx.framebuffer(
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

    def __del__(self) -> None:
        if self.__vao is not None:
            self.__vao.release()
            del self.__vao

        if self.__vbo is not None:
            self.__vbo.release()
            del self.__vbo

        if self.__fbo is not None:
            self.__fbo.release()
            del self.__fbo

        if self.__texture is not None:
            self.__texture.release()
            del self.__texture

        if self.__ctx is not None:
            # logger.debug("clean")
            self.__ctx.release()
            self.__ctx.gc()
            del self.__ctx

    def reset(self) -> None:
        self.__runtime = 0
        self.__delta = 0
        self.__frame_count = 0
        self.__time_last = time.perf_counter()

    def __bufferReset(self) -> None:
        if self.__fbo is not None:
            self.__fbo.release()
            del self.__fbo
            self.__fbo = None

        if self.__texture is not None:
            self.__texture.release()
            del self.__texture
            self.__texture = None

        self.__texture = self.__ctx.texture((self.__width, self.__height), 3)
        try:
            self.__fbo = self.__ctx.framebuffer(
                color_attachments=[self.__texture]
            )
        except Exception as e:
            logger.error(str(e))
            logger.debug(self.__ctx)
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
        self.__width = max(0, min(val, MAX_WIDTH))
        self.__bufferReset()

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, val: int) -> None:
        self.__height = max(0, min(val, MAX_HEIGHT))
        self.__bufferReset()

    def __set_uniforms(self, channel0: Image=None) -> None:
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

        if self.__iChannel0 is not None and channel0 is not None:
            if len(channel0.mode) == 4:
                channel0 = channel0.convert("RGB")
            texture = self.__ctx.texture(channel0.size, components=3, data=channel0.tobytes())
            texture.use(location=0)

    def render(self, channel0:Image=None, param:dict=None) -> Image:
        self.__fbo.use()
        self.__fbo.clear(0.0, 0.0, 0.0)
        if not self.__hold:
            self.__set_uniforms(channel0)
            # logger.debug(self.__param)
            # logger.debug(param)
            for k, v in (param or {}).items():
                try:
                    self.__param[k].value = v
                except KeyError as _:
                    pass
                except Exception as e:
                    logger.error(k, v)
                    logger.error(str(e))

        self.__vao.render()
        pixels = self.__fbo.color_attachments[0].read()
        self.__frame = Image.frombytes(
            "RGB", self.__fbo.size, pixels,
            "raw", "RGB", 0, -1
        )
        self.__frame = self.__frame.transpose(Image.FLIP_TOP_BOTTOM)

        # step frame
        if not self.__hold:
            self.__frame_count += 1
            self.__delta = max(0, self.__fps_rate) if self.__fps > 0 else time.perf_counter() - self.__time_last
            self.__runtime += self.__delta
            self.__time_last = time.perf_counter()

        return self.__frame
