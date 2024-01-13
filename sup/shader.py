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

from Jovimetrix.sup.image import image_save_gif

# =============================================================================

MAX_WIDTH = 8192
MAX_HEIGHT = 8192

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

uniform float iUser1;
uniform float iUser2;

#define texture2D texture
layout(location = 0) out vec4 fragColor;
"""

# =============================================================================

class CompileException(Exception): pass

# =============================================================================

class GLSL:
    def __init__(self, fragment:str, width:int=128, height:int=128) -> None:
        self.__fragment = FRAGMENT_HEADER + fragment
        self.__ctx = moderngl.create_standalone_context()

        try:
            self.__prog = self.__ctx.program(
                vertex_shader=VERTEX,
                fragment_shader=self.__fragment,
            )
        except Exception as e:
            raise CompileException(e)

        self.__iResolution = self.__prog.get('iResolution', None)
        self.__iTime = self.__prog.get('iTime', None)
        self.__iTimeDelta = self.__prog.get('iTimeDelta', None)
        self.__iFrameRate = self.__prog.get('iFrameRate', None)
        self.__iFrame = self.__prog.get('iFrame', None)

        try:
            self.__prog['iChannel0'].value = 0
        except:
            pass
        self.__iChannel0 = self.__prog.get('iChannel0', None)

        try:
            self.__prog['iChannel1'].value = 0
        except:
            pass
        self.__iChannel1 = self.__prog.get('iChannel1', None)

        self.__iUser1 = self.__prog.get('iUser1', None)
        self.__iUser2 = self.__prog.get('iUser2', None)

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
        self.__fbo = self.__ctx.framebuffer(
            color_attachments=[self.__ctx.texture((self.__width, self.__height), 3)]
        )

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, val: int) -> None:
        self.__height = max(0, min(val, MAX_HEIGHT))
        self.__fbo = self.__ctx.framebuffer(
            color_attachments=[self.__ctx.texture((self.__width, self.__height), 3)]
        )

    def __set_uniforms(self, channel0: Image=None, channel1: Image=None, user1:float=None, user2:float=None) -> None:
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
            if (size := len(channel0.mode)) == 4:
                channel0 = channel0.convert("RGB")
            texture: Image = self.__ctx.texture(channel0.size, components=size, data=channel0.tobytes())
            texture.use(location=0)

        if self.__iChannel1 is not None and channel1 is not None:
            if (size := len(channel1.mode)) == 4:
                channel1 = channel1.convert("RGB")
            texture: Image = self.__ctx.texture(channel1.size, components=size, data=channel1.tobytes())
            texture.use(location=1)

        if self.__iUser1 is not None and user1 is not None:
            self.__iUser1.value = user1

        if self.__iUser2 is not None and user2 is not None:
            self.__iUser2.value = user2

    def render(self, channel0:Image=None, channel1:Image=None, user1:float=None, user2:float=None) -> None:
        self.__fbo.clear(0.0, 0.0, 0.0)
        self.__fbo.use()
        if not self.__hold:
            self.__set_uniforms(channel0, channel1, user1, user2)

        self.__vao.render()
        self.__frame = Image.frombytes(
            "RGB", self.__fbo.size, self.__fbo.color_attachments[0].read(),
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
