"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
GLSL Support
"""

import os
import time
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
from PIL import Image
from loguru import logger

from Jovimetrix.sup.image import image_save_gif

# =============================================================================

VERTEX = """
#version 330
in vec2 iPosition;
out vec2 iUV;

void main() {
    gl_Position = vec4(iPosition, 0.0, 1.0);
    iUV = iPosition / 2.0 + 0.5;
}"""

FRAGMENT_HEADER = """
#version 330
precision highp float;

in vec2 iUV;
uniform vec2 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrameRate;
uniform int iFrame;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;

uniform float iUser0;
uniform float iUser1;

#define texture2D texture
layout(location = 0) out vec4 fragColor;
"""

class GLSL:
    def __init__(self, fragment:str, width:int=128, height:int=128) -> None:
        self.__fragment = FRAGMENT_HEADER + fragment
        self.__ctx = moderngl.create_standalone_context()
        try:
            self.__prog = self.__ctx.program(
                vertex_shader=VERTEX,
                fragment_shader=self.__fragment,
            )
        except:
            raise Exception(self.__fragment)

        self.__iResolution = self.__prog.get('iResolution', None)
        self.__iTime = self.__prog.get('iTime', None)
        self.__iDelta = self.__prog.get('iDelta', None)
        self.__iFrameRate = self.__prog.get('iFrameRate', None)
        self.__iFrame = self.__prog.get('iFrame', None)

        self.__iChannel0 = self.__prog.get('iChannel0', None)
        self.__iChannel1 = self.__prog.get('iChannel1', None)
        self.__iUser0 = self.__prog.get('iUser0', None)
        self.__iUser1 = self.__prog.get('iUser1', None)

        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype='f4')

        self.__width = width
        self.__height = height
        self.__vbo = self.__ctx.buffer(vertices.tobytes())
        self.__vao = self.__ctx.simple_vertex_array(self.__prog, self.__vbo, "iPosition")
        self.__fbo = self.__ctx.framebuffer(
            color_attachments=[self.__ctx.texture((width, height), 3)]
        )

        self.__runtime: float = 0
        self.__delta: float = 0
        self.__frame_count: int = 0
        # FPS > 0 will act as a step (per frame step)
        self.__fps: float = 0
        self.__fps_rate: float = 0
        # the last frame rendered
        self.__frame: Image = Image.new("RGB", (0, 0))
        self.__hold: bool = False
        self.__time_last: float = time.perf_counter()

    def reset(self) -> None:
        if self.__iFrame is not None:
            self.__iFrame = 0

        self.__runtime = 0
        self.__delta = 0
        self.__frame_count = 0
        self.__time_last = time.perf_counter()

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
    def fps(self) -> int:
        return self.__fps

    @fps.setter
    def runtime(self, val:int) -> None:
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
        self.__width = max(0, min(val, 4096))
        self.__fbo = self.__ctx.framebuffer(
            color_attachments=[self.__ctx.texture((self.__width, self.__height), 3)]
        )

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, val: int) -> None:
        self.__height = max(0, min(val, 4096))
        self.__fbo = self.__ctx.framebuffer(
            color_attachments=[self.__ctx.texture((self.__width, self.__height), 3)]
        )

    def __set_uniforms(self, user0: Any = None, user1: Any = None) -> None:
        if self.__iResolution is not None:
            self.__iResolution.value = (self.__width, self.__height)

        if self.__iTime is not None:
            self.__iTime.value = self.__runtime

        if self.__iDelta is not None:
            self.__iDelta.value = self.__delta

        if self.__iFrameRate is not None:
            self.__iFrameRate.value = self.__fps_rate

        if self.__iFrame is not None:
            self.__iFrame.value = self.__frame_count

        if user0 is not None:
            self.__iUser0.value = user0

        if user1 is not None:
            self.__iUser1.value = user1

    def render(self, user0: Any = None, user1: Any = None) -> None:
        if self.__hold:
            return self.__frame

        self.__fbo.clear(0.0, 0.0, 0.0)
        self.__fbo.use()
        self.__set_uniforms(user0, user1)
        self.__vao.render()
        self.__frame = Image.frombytes(
            "RGB", self.__fbo.size, self.__fbo.color_attachments[0].read(),
            "raw", "RGB", 0, -1
        )
        self.__frame_count += 1
        self.__delta = max(0, self.__fps_rate) if self.__fps > 0 else time.perf_counter() - self.__time_last
        self.__runtime += self.__delta
        self.__time_last = time.perf_counter()
        return self.__frame

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    fragment = r"C:\dev\ComfyUI\ComfyUI\custom_nodes\Jovimetrix\glsl\gradient.glsl"
    glsl = GLSL(fragment, 256, 256)
    glsl.fps = 60
    images = [glsl.render() for _ in range(120)]
    root = os.path.dirname(__file__)
    image_save_gif(root + f"/../_res/tst/glsl.gif", images, glsl.fps)
    for i, x in enumerate(images):
        x.save( root + f"/../_res/tst/glsl-{i}.gif")
    print(Image.open(root + f"/../_res/tst/glsl.gif").n_frames)
