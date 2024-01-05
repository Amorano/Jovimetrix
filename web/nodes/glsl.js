/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js"
import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from '../core/util.js'
import { ComfyWidgets } from "/scripts/widgets.js"
import { SpinnerWidget } from "../widget/widget_spinner.js"

const VERTEX_SHADER = `#version 300 es
in vec2 iResolution;
in vec2 iPosition;
out vec2 iCoord;
void main() {
    gl_Position = vec4(iPosition, 0.0, 1.0);
    iCoord = iPosition * 0.5 + 0.5;
}`

// vec4 color = texture(iChannel0, iCoord); color.r += sin(iTime);
// vec4(clamp(color - colorShift, 0.0, 1.0), 1.0);

const FRAGMENT_DEFAULT2 = `void main() {
    vec4 color = texture(iChannel0, iCoord);
    color.r += sin(iTime);
    FragColor = color;
}`

const FRAGMENT_DEFAULT = `void main() {
    vec4 color = texture(iChannel0, iCoord);
    color.r += cos(iTime / 5.0);
    color.g += sin(iTime / 4.0);
    color.b += cos(iTime / 3.0) + sin(iTime / 2.0);
    FragColor = color;
}`



const FRAGMENT_HEADER = (body) => {
    return `#version 300 es
precision highp float;
in vec2 iCoord;
uniform sampler2D iChannel0;
uniform float iTime;
out vec4 FragColor;

` + body};

function get_position_style(ctx, widget_width, y, node_height) {
    const MARGIN = 4;
    const elRect = ctx.canvas.getBoundingClientRect();
    const transform = new DOMMatrix()
        .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
        .multiplySelf(ctx.getTransform())
        .translateSelf(MARGIN, MARGIN + y);

    return {
        transformOrigin: '0 0',
        transform: transform,
        left: `0px`,
        top: `0px`,
        position: "absolute",
        maxWidth: `${widget_width - MARGIN * 2}px`,
        maxHeight: `${node_height - MARGIN * 2}px`,
        width: `${ctx.canvas.width}px`,  // Set canvas width
        height: `${ctx.canvas.height}px`,  // Set canvas height
    };
}

const _id = "GLSL (JOV) 🍩"
const GLSLWidget = (app, inputName, fragment) => {

    const canvas = $el("canvas")
    canvas.style.backgroundColor = "rgba(0, 0, 0, 1)"
    canvas.width = 512
    canvas.height = 512
    const GL = canvas.getContext('webgl2');
    let PROGRAM;

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
        inputEl: canvas,
        FRAGMENT: fragment,
        compiled: false,
        initShaderProgram() {
            const vertex = compileShader(VERTEX_SHADER, GL.VERTEX_SHADER);
            const fragment_full = FRAGMENT_HEADER(this.FRAGMENT);
            const fragment = compileShader(fragment_full, GL.FRAGMENT_SHADER);

            if (!vertex || !fragment) {
                console.error(GL.getShaderInfoLog(vertex));
                console.error(GL.getShaderInfoLog(fragment));
                this.compiled = false;
                return null;
            }

            PROGRAM = GL.createProgram();
            GL.attachShader(PROGRAM, vertex);
            GL.attachShader(PROGRAM, fragment);
            GL.linkProgram(PROGRAM);

            if (!GL.getProgramParameter(PROGRAM, GL.LINK_STATUS)) {
                console.error('Unable to initialize the shader program: ' + GL.getProgramInfoLog(PROGRAM));
                console.error(GL.getShaderInfoLog(vertex));
                console.error(GL.getShaderInfoLog(fragment));
                console.error(GL.getProgramInfoLog(PROGRAM));
                this.compiled = false;
                return null;
            }

            GL.useProgram(PROGRAM);
            this.compiled = true;
            // console.debug(fragment_full)

            const positionBuffer = GL.createBuffer();
            GL.bindBuffer(GL.ARRAY_BUFFER, positionBuffer);
            GL.bufferData(GL.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1]), GL.STATIC_DRAW);

            const positionAttr = GL.getAttribLocation(PROGRAM, 'iPosition');
            GL.vertexAttribPointer(positionAttr, 2, GL.FLOAT, false, 0, 0);
            GL.enableVertexAttribArray(positionAttr);

            // Set the initial resolution
            this.update_resolution(this.inputEl.width, this.inputEl.height);
        },

        render() {
            GL.clearColor(0, 0, 0, 1);
            GL.clear(GL.COLOR_BUFFER_BIT);
            if (PROGRAM === undefined && this.FRAGMENT != undefined) {
                this.initShaderProgram();
            }
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
            const loc = GL.getUniformLocation(PROGRAM, "iResolution");
            GL.uniform2f(loc, width, height);
        },

        draw(ctx, node, widget_width, y, widget_height) {
            // assign the required style when we are drawn
            Object.assign(this.inputEl.style, get_position_style(ctx, widget_width, y, node.size[1]));
            // this.render();
            //this.value = this.inputEl.innerHTML
        },
        mouse(e, pos, node) {
            if (e.type === 'pointermove') {
                console.debug(e.delta);
            }
        },
        computeSize(width) {
            return [width, LiteGraph.NODE_WIDGET_HEIGHT]
        },
        async serializeValue(nodeId, widgetIndex) {
            if (widgetIndex !== 4) {
                return;
            }

            const pixels = new Uint8Array(canvas.width * canvas.height * 4);
            GL.readPixels(0, 0, canvas.width, canvas.height, GL.RGBA, GL.UNSIGNED_BYTE, pixels);

            const img = new Image();
            img.src = canvas.toDataURL(); // Convert canvas to data URL

            return new Promise((resolve) => {
                img.onload = function () {
                    const tempCanvas = document.createElement('canvas');
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCanvas.width = img.width;
                    tempCanvas.height = img.height;
                    tempCtx.drawImage(img, 0, 0);
                    const base64String = tempCanvas.toDataURL('image/png').split(',')[1];
                    console.info(base64String)
                    resolve(base64String);
                };
            });
        }
    }
    document.body.appendChild(widget.inputEl);
    return widget
};

const ext = {
	name: _id,
    async getCustomWidgets(app) {
        return {
            GLSL: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(GLSLWidget(app, inputName, inputData)),
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

            this.widget_time = ComfyWidgets.FLOAT(this, '🕛', ["FLOAT", {"default": 0, "step": 0.01, "min": 0}], app).widget
            this.widget_fixed = ComfyWidgets.FLOAT(this, 'FIXED', ["FLOAT", {"default": 0, "step": 0.01, "min": 0}], app).widget
            this.widget_reset = ComfyWidgets.BOOLEAN(this, 'RESET', ["BOOLEAN", {"default": false}], app).widget
            this.widget_wh = this.addCustomWidget(SpinnerWidget(app, "WH", ["VEC2", {"default": [512, 512], "step": 1, "min": 1}], [512, 512]))
            this.widget_fragment = ComfyWidgets.STRING(this, FRAGMENT_DEFAULT, ["STRING", {multiline: true}], app).widget
            const widget_glsl = this.addCustomWidget(GLSLWidget(app, 'GLSL', FRAGMENT_DEFAULT))
            widget_glsl.render()

            let TIME = 0
            const onExecutionStart = nodeType.prototype.onExecutionStart;
            nodeType.prototype.onExecutionStart = function (message) {
                onExecutionStart?.apply(this, arguments);
                if (TIME == 0) {
                    this.widget_time.value = 0
                }
                if (this.widget_reset.value == false) {
                    this.widget_time.value += (performance.now() - TIME)  / 1000;
                } else {
                    TIME = 0;
                    this.widget_time.value = 0;
                }
                TIME = performance.now()
                if (this.inputs && this.inputs[0].value !== undefined) {
                    console.debug("GLSL", this.inputs[0].value, this.inputs)
                    widget_glsl.update_texture(0, this.inputs[0].value);
                }
                widget_glsl.update_resolution(this.widget_wh.value[0], this.widget_wh.value[1]);
                widget_glsl.update_time(this.widget_time.value)
                widget_glsl.render();
            };

            function python_grab_image(event) {
                console.info(event)
            }
            api.addEventListener("jovi-glsl-image", python_grab_image);

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function (message) {
                onRemoved?.apply(this, arguments);
                widget_glsl.inputEl.remove();
                util.cleanupNode(this);
            };

            //this.setOutputData('🖼️', 0)
            this.widget_image = this.addOutput("🖼️", 'IMAGE');
            this.widget_mask = this.addOutput("😷", 'MASK');

            return me;
        }
    }
}

app.registerExtension(ext)
