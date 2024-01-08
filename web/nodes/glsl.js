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
in vec2 iResolution;
in vec2 iPosition;
out vec2 iUV;
void main() {
    gl_Position = vec4(iPosition, 0.0, 1.0);
    iUV = iPosition;
}`

const FRAGMENT_HEADER = (body) => {
    return `#version 300 es
#ifdef GL_ES
precision mediump float;
#endif

#define PI 3.14159265359

precision highp float;
in vec2 iUV;
uniform vec2 iResolution;
uniform sampler2D iChannel0;
uniform float iTime;
uniform float iUser1;
uniform float iUser2;
out vec4 fragColor;

` + body};

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

    let CANVAS = new OffscreenCanvas(width, height);
    let GL = CANVAS.getContext('webgl2');
    let PROGRAM;
    let BUFFER_POSITION;

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

            GL.viewport(0, 0, CANVAS.width, CANVAS.height);
            GL.scissor(0, 0, CANVAS.width, CANVAS.height);
            GL.useProgram(PROGRAM);
            this.compiled = true;
        },

        render() {
            GL.clearColor(0, 0, 0, 1);
            GL.clear(GL.COLOR_BUFFER_BIT);
            if (PROGRAM === undefined) {
                if (this.FRAGMENT != undefined) {
                    // console.info('init')
                    this.initShaderProgram();
                }
            }
            //GL.viewport(0, 0, CANVAS.width, CANVAS.height);
            //GL.scissor(0, 0, CANVAS.width, CANVAS.height);
            GL.drawArrays(GL.TRIANGLES, 0, 6);
        },

        update_texture(index, texture) {
            GL.activeTexture(GL.TEXTURE0 + index);
            GL.bindTexture(GL.TEXTURE_2D, texture);
            const loc = GL.getUniformLocation(PROGRAM, "iChannel" + index);
            GL.uniform1i(loc, index);
        },

        update_time(time) {
            const loc = GL.getUniformLocation(PROGRAM, "iTime");
            GL.uniform1f(loc, time);
        },

        update_resolution(width, height) {
            if (width != CANVAS.width || height != CANVAS.height) {
                CANVAS.width = width;
                CANVAS.height = height;
                GL.viewport(0, 0, CANVAS.width, CANVAS.height);
                GL.scissor(0, 0, CANVAS.width, CANVAS.height);
                const loc = GL.getUniformLocation(PROGRAM, "iResolution");
                GL.uniform2f(loc, width, height);
            }
        },

        update_user1(user) {
            const loc = GL.getUniformLocation(PROGRAM, "iUser1");
            GL.uniform1f(loc, user);
            // console.info(1, user);
        },

        update_user2(user) {
            const loc = GL.getUniformLocation(PROGRAM, "iUser2");
            GL.uniform1f(loc, user);
            //console.info(2, user);
        },

        async frame() {
            const pixels = new Uint8Array(CANVAS.width * CANVAS.height * 4);
            GL.readPixels(0, 0, CANVAS.width, CANVAS.height, GL.RGBA, GL.UNSIGNED_BYTE, pixels);
            const imageData = new ImageData(new Uint8ClampedArray(pixels), CANVAS.width, CANVAS.height);
            const imageBitmap = await createImageBitmap(imageData);
            CANVAS_TEMP.width = CANVAS.width;
            CANVAS_TEMP.height = CANVAS.height;
            CANVAS_TEMP_CTX.drawImage(imageBitmap, 0, 0);
            const data = CANVAS_TEMP.toDataURL('image/png').split(',')[1];
            //console.info(data)
            return data;
        },

        async serializeValue(nodeId, widgetIndex) {
            if (widgetIndex !== 7) {
                return;
            }
            // console.info(nodeId);
            return await this.frame();
        },

        /*
        draw(ctx, node, widget_width, y, widget_height) {
            Object.assign(CANVAS_TEMP.style, get_position_style(ctx, widget_width, y, CANVAS.height));
        },
        */

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
            const widget_fixed = this.widgets[1];
            const widget_reset = this.widgets[2];
            const widget_wh = this.widgets[3];
            const widget_fragment = this.widgets[4];
            const widget_user1 = this.widgets[5];
            const widget_user2 = this.widgets[6];
            widget_fragment.inputEl.addEventListener('input', function (event) {
                // console.log('Textarea content changed:', event.target.value);
                widget_glsl.FRAGMENT = event.target.value;
                widget_glsl.initShaderProgram();
            });
            const widget_glsl = this.addCustomWidget(GLSLWidget('GLSL', widget_fragment.value, widget_wh.value[0], widget_wh.value[1]))
            widget_glsl.render()

            let TIME = 0;
            const onExecutionStart = nodeType.prototype.onExecutionStart;
            nodeType.prototype.onExecutionStart = function (message) {
                onExecutionStart?.apply(this, arguments);

                if (TIME == 0) {
                    widget_time.value = 0
                }

                if (widget_reset.value == false) {
                    if (widget_fixed.value > 0) {
                        widget_time.value += widget_fixed.value;
                    } else {
                        widget_time.value += (performance.now() - TIME)  / 1000;
                    }
                } else {
                    TIME = 0;
                    widget_time.value = 0;
                }

                TIME = performance.now()
                // console.info(this)
                if (this.inputs && this.inputs[0].value !== undefined) {
                    // console.debug("GLSL", this.inputs[0].value, this.inputs)
                    widget_glsl.update_texture(0, this.inputs[0].value);
                    // console.info(this.inputs[0].value)
                }
                widget_glsl.update_time(widget_time.value);
                widget_glsl.update_resolution(widget_wh.value[0], widget_wh.value[1]);
                widget_glsl.update_user1(widget_user1.value);
                widget_glsl.update_user2(widget_user2.value);
                widget_glsl.render();
                app.canvas.setDirty(true)
            };

            async function python_grab_image(event) {
                const frame = await widget_glsl.frame();
                var data = { id: event.detail.id, frame: frame }
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
