/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js"
import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from '../core/util.js'

const VERTEX_SHADER = `#version 300 es
in vec2 iPosition;
out vec2 iUV;
void main()
{
    vec2 verts[6] = vec2[](
        vec2(-1, -1), vec2(1, -1), vec2(-1, 1),
        vec2(1, -1), vec2(1, 1), vec2(-1, 1)
    );
    gl_Position = vec4(verts[gl_VertexID], 0, 1);
    iUV = iPosition;
}`

const FRAGMENT_HEADER = (body) => {
    return `#version 300 es
precision highp float;
in vec2 iUV;
uniform vec2	iResolution;
uniform float	iTime;
uniform float	iDelta;
uniform float	iFrameRate;
uniform int	    iFrame;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;

uniform float iUser0;
uniform float iUser1;

#define texture2D texture
out vec4 fragColor;`
 + body};

function get_position_style(ctx, width, y, height) {
    const MARGIN = 4;
    const elRect = ctx.canvas.getBoundingClientRect();
    const transform = new DOMMatrix()
        .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
        .multiplySelf(ctx.getTransform())
        .translateSelf(MARGIN, MARGIN + y);

    return {
        transformOrigin: '0 0',
        transform: transform,
        left: `4px`,
        top: `0px`,
        position: "absolute",
        maxWidth: `${width - MARGIN * 2}px`,
        maxHeight: `${height - MARGIN * 2}px`,
        width: width - MARGIN * 2,
        height: height - MARGIN * 2
    };
}

const _id = "GLSL (JOV) 🍩"
const GLSLWidget = (inputName, fragment, width=512, height=512) => {

    const CANVAS_TEMP = document.createElement('canvas');
    // document.body.appendChild(CANVAS_TEMP);
    // const CANVAS_TEMP = new OffscreenCanvas(width, height);
    const CANVAS_TEMP_CTX = CANVAS_TEMP.getContext('2d');

    const CANVAS = new OffscreenCanvas(width, height);
    const GL = CANVAS.getContext('webgl2');
    let PROGRAM;
    let BUFFER_POSITION;
    let FBO;
    let TEXTURE;
    let TEXTURES;

    function compileShader (source, type) {
        const shader = GL.createShader(type);
        GL.shaderSource(shader, source);
        GL.compileShader(shader);

        if (!GL.getShaderParameter(shader, GL.COMPILE_STATUS)) {
            console.error('Shader compilation error: ' + GL.getShaderInfoLog(shader));
            GL.deleteShader(shader);
            return null;
        }
        return shader;
    };

    const widget = {
        type: 'GLSL',
        name: inputName,
        y: 0,
        FRAGMENT: fragment,
        compiled: false,
        canvas: CANVAS,
        initShaderProgram() {
            const vertex_shader = compileShader(VERTEX_SHADER, GL.VERTEX_SHADER);
            const fragment_full = FRAGMENT_HEADER(this.FRAGMENT);
            const fragment_shader = compileShader(fragment_full, GL.FRAGMENT_SHADER);

            if (!fragment_shader) {
                console.error(GL.getShaderInfoLog(fragment_shader));
                this.compiled = false;
                return null;
            }

            if (PROGRAM) {
                GL.deleteProgram(PROGRAM);
            }
            PROGRAM = GL.createProgram();
            GL.attachShader(PROGRAM, vertex_shader);
            GL.attachShader(PROGRAM, fragment_shader);
            GL.linkProgram(PROGRAM);

            if (!GL.getProgramParameter(PROGRAM, GL.LINK_STATUS)) {
                console.error('Unable to initialize the shader program: ' + GL.getProgramInfoLog(PROGRAM));
                console.error(GL.getShaderInfoLog(vertex_shader));
                console.error(GL.getShaderInfoLog(fragment_shader));
                console.error(GL.getProgramInfoLog(PROGRAM));
                cleanup();
                return null;
            }

            GL.detachShader(PROGRAM, vertex_shader);
            GL.detachShader(PROGRAM, fragment_shader);
            GL.deleteShader(vertex_shader);
            GL.deleteShader(fragment_shader);

            BUFFER_POSITION = GL.createBuffer();
            GL.bindBuffer(GL.ARRAY_BUFFER, BUFFER_POSITION);
            GL.bufferData(GL.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1]), GL.STATIC_DRAW);

            const positionAttr = GL.getAttribLocation(PROGRAM, 'iPosition');
            GL.vertexAttribPointer(positionAttr, 2, GL.FLOAT, false, 0, 0);
            GL.enableVertexAttribArray(positionAttr);


            TEXTURE = GL.createTexture();
            GL.bindTexture(GL.TEXTURE_2D, TEXTURE);
            GL.texImage2D(GL.TEXTURE_2D, 0, GL.RGB, width, height, 0, GL.RGB, GL.UNSIGNED_BYTE, null);
            GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MIN_FILTER, GL.LINEAR);
            GL.texParameteri(GL.TEXTURE_2D, GL.TEXTURE_MAG_FILTER, GL.LINEAR);

            /*
            FBO = GL.createFramebuffer();
            GL.bindFramebuffer(GL.FRAMEBUFFER, FBO);
            GL.framebufferTexture2D(GL.FRAMEBUFFER, GL.COLOR_ATTACHMENT0, GL.TEXTURE_2D, TEXTURE, 0);
            if (GL.checkFramebufferStatus(GL.FRAMEBUFFER) != GL.FRAMEBUFFER_COMPLETE) {
                console.error("Framebuffer is not complete");
            }*/

            TEXTURES = [];
            for (let i = 0; i < 4; i++) {
                const texture = GL.createTexture();
                GL.bindTexture(GL.TEXTURE_2D, texture);
                // Set texture parameters and data if needed
                TEXTURES.push(texture);
            }

            GL.useProgram(PROGRAM);
            this.compiled = true;
        },

        render() {
            GL.bindFramebuffer(GL.GL_FRAMEBUFFER, FBO)
            GL.clearColor(0, 0, 0, 1);
            GL.clear(GL.COLOR_BUFFER_BIT);
            if (PROGRAM === undefined) {
                if (this.FRAGMENT != undefined) {
                    this.initShaderProgram();
                }
            }
            GL.drawArrays(GL.TRIANGLES, 0, 6);
        },
/*
        uniform vec3	iResolution;
        uniform float	iTime;
        uniform float	iDelta;
        uniform float	iFrameRate;
        uniform int	    iFrame;

        uniform sampler2D iChannel0;
        uniform sampler2D iChannel1;

        uniform float iUser0;
        uniform float iUser1;
*/
        update_resolution(width, height) {
            CANVAS.width = width;
            CANVAS.height = height;
            GL.viewport(0, 0, CANVAS.width, CANVAS.height);
            const loc = GL.getUniformLocation(PROGRAM, "iResolution");
            GL.uniform2f(loc, width, height);
            // console.log("resolution updated", width, height)
        },

        update_texture(texture, index) {
            GL.activeTexture(GL.TEXTURE0 + index);
            GL.bindTexture(GL.TEXTURE_2D, texture);
            loc = GL.getUniformLocation(PROGRAM, "iChannel" + index);
            GL.uniform1i(loc, index);
            console.log("texture updated", index)
        },

        async frame(time, delta, frame, user1, user2) {
            let loc;
            loc = GL.getUniformLocation(PROGRAM, "iTime");
            GL.uniform1f(loc, time);

            loc = GL.getUniformLocation(PROGRAM, "iDelta");
            GL.uniform1f(loc, delta);

            loc = GL.getUniformLocation(PROGRAM, "iFrame");
            GL.uniform1i(loc, frame);

            loc = GL.getUniformLocation(PROGRAM, "iUser1");
            GL.uniform1f(loc, user1);

            loc = GL.getUniformLocation(PROGRAM, "iUser2");
            GL.uniform1f(loc, user2);

            const pixels = new Uint8Array(CANVAS.width * CANVAS.height * 4);
            GL.readPixels(0, 0, CANVAS.width, CANVAS.height, GL.RGBA, GL.UNSIGNED_BYTE, pixels);
            const imageData = new ImageData(new Uint8ClampedArray(pixels), CANVAS.width, CANVAS.height);
            const imageBitmap = await createImageBitmap(imageData);
            CANVAS_TEMP.width = CANVAS.width;
            CANVAS_TEMP.height = CANVAS.height;
            CANVAS_TEMP_CTX.drawImage(imageBitmap, 0, 0);
            return CANVAS_TEMP.toDataURL('image/png').split(',')[1];
        },

        cleanup() {
            this.compiled = false;
            GL.useProgram(null);
            if (CANVAS_TEMP) {
                CANVAS_TEMP.remove();
            }
            if (BUFFER_POSITION) {
                GL.deleteBuffer(BUFFER_POSITION);
            }
            if (PROGRAM) {
                GL.deleteProgram(PROGRAM);
            }
            CANVAS.close();
        },
    }
    //document.body.appendChild(CANVAS_TEMP);
    return widget
};

const ext = {
	name: _id,
    async getCustomWidgets(app) {
        return {
            GLSL: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(GLSLWidget(inputName, inputData)),
            }),
        }
    },
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const widget_time = this.widgets[0];
            const widget_fps = this.widgets[1];
            const widget_batch = this.widgets[2];
            const widget_reset = this.widgets[3];
            const widget_wh = this.widgets[4];
            const widget_fragment = this.widgets[5];
            const widget_user1 = this.widgets[6];
            const widget_user2 = this.widgets[7];
            widget_fragment.inputEl.addEventListener('input', function (event) {
                widget_glsl.FRAGMENT = event.target.value;
                widget_glsl.initShaderProgram();
            });
            const widget_glsl = this.addCustomWidget(GLSLWidget('GLSL', widget_fragment.value, widget_wh.value[0], widget_wh.value[1]))
            let frame_count = 0;

            let TIME = 0;
            async function python_grab_image(event) {
                let frames = [];
                for (let i = 0; i < widget_batch.value; i++) {
                    widget_glsl.render();
                    let delta = 0;
                    if (widget_reset.value == false) {
                        if (widget_fps.value > 0) {
                            widget_time.value += (1.0 / widget_fps.value);
                        } else {
                            delta = (performance.now() - TIME)  / 1000.0;
                            widget_time.value += delta;
                        }
                    } else {
                        TIME = 0;
                        widget_time.value = 0;
                        frame_count = 0;
                    }
                    TIME = performance.now();
                    if (this.inputs && this.inputs[0].value !== undefined) {
                        widget_glsl.update_texture(this.inputs[0].value, 0)
                    }
                    if (this.inputs && this.inputs[1].value !== undefined) {
                        widget_glsl.update_texture(this.inputs[1].value, 1)
                    }

                    widget_glsl.update_resolution(widget_wh.value[0], widget_wh.value[1]);
                    const frame = await widget_glsl.frame(widget_time.value, delta, frame_count, widget_user1.value, widget_user2.value);

                    if (widget_reset.value == false) {
                        frames.push(frame);
                        frame_count++;
                    } else {
                        frames = frames.concat(Array(widget_batch.value).fill(frame))
                        break;
                    }
                };
                var data = { id: event.detail.id, frame: frames }
                util.api_post('/jovimetrix/message', data);
            }
            api.addEventListener("jovi-glsl-image", python_grab_image);

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function (message) {
                onRemoved?.apply(this, arguments);
                widget_glsl.cleanup();
                util.cleanupNode(this);
            };

            this.serialize_widgets = true;
            return me;
        }
    }
}

app.registerExtension(ext)
