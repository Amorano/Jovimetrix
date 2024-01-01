/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"

const VERTEX_SHADER = `#version 300 es
in vec2 iResolution;
in vec2 iPosition;
out vec2 iCoord;
void main() {
    gl_Position = vec4(iPosition, 0.0, 1.0);
    iCoord = iPosition * 0.5 + 0.5;
}`

// vec4 color = texture(iChannel0, iCoord); color.r += sin(iTime);

const FRAGMENT_DEFAULT = `void main() {
    vec4 color = texture(iChannel0, iCoord);
    color.r += sin(iTime);
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
const GLSLWidget = (app, inputName, inputData) => {

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
        FRAGMENT: inputData.input?.optional?.FRAGMENT[1].default || FRAGMENT_DEFAULT,
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
                    resolve(base64String);
                };
            });
        }
    }
    document.body.appendChild(widget.inputEl);
    return widget
};

const ext = {
	name: 'jovimetrix.node.glsl',
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
            const widget_glsl = this.addCustomWidget(GLSLWidget(app, 'GLSL', nodeData))
            widget_glsl.render()
            //this.setSize([this.size[0], this.computeSize()[1] + widget_glsl.inputEl.offsetHeight])
            // 🕛 🎬
            let TIME = 0
            const onExecutionStart = nodeType.prototype.onExecutionStart;
            nodeType.prototype.onExecutionStart = function (message) {
                onExecutionStart?.apply(this, arguments);
                // check if the fragment shader changed...
                /*
                const short = this.inputs.optional;
                console.debug(this.inputs)
                if (!widget_glsl.compiled || short.FRAGMENT != widget_glsl.FRAGMENT) {
                    TIME = 0;
                    widget_glsl.FRAGMENT = short.FRAGMENT[1].value;
                    widget_glsl.initShaderProgram()
                }
                */
                if (TIME == 0) {
                    this.widgets[1].value = 0
                    TIME = performance.now();
                }
                this.widgets[1].value += (performance.now() - TIME)  / 1000;
                TIME = performance.now()
                console.debug(this)
                if (this.inputs[0].value !== undefined) {
                    console.debug("GLSL", this.inputs[0].value, this.inputs)
                    widget_glsl.update_texture(0, this.widgets[1].value);
                }
                widget_glsl.update_resolution(this.widgets[3].value[0], this.widgets[3].value[1]);
                widget_glsl.update_time(this.widgets[1].value)
                widget_glsl.render();
                /*
                this.setOutputData('🖼️', 0)
                this.setOutputData('😷', 0)*/
            }

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function (message) {
                onRemoved?.apply(this, arguments);
                widget_glsl.inputEl.remove();
                util.cleanupNode(this);
            };
            return me;
        }
    }
}

app.registerExtension(ext)
