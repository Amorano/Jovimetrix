/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"

const VERTEX_SHADER = `#version 300 es
in vec2 position;
out vec2 texCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    texCoord = position * 0.5 + 0.5;
}
`

const FRAGMENT_DEFAULT = `vec4 color = texture(textureSampler, texCoord);
    color.r += sin(time);
    FragColor = color;`;

const FRAGMENT_HEADER = (body) => {
    return `#version 300 es
precision highp float;
in vec2 texCoord;
uniform sampler2D textureSampler;
uniform float time;
out vec4 FragColor;

void main() {
` + body + `}`
}


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
        position: "relative",
        maxWidth: `${widget_width - MARGIN * 2}px`,
        maxHeight: `${node_height - MARGIN * 2}px`,
        width: `${ctx.canvas.width}px`,  // Set canvas width
        height: `${ctx.canvas.height}px`,  // Set canvas height
    };
}

const _id = "GLSL (JOV) 🍩"

const GLSLWidget = (app, inputData) => {

    const canvas = $el("canvas")
    canvas.style.backgroundColor = "rgba(0, 0, 0, 1)"
    canvas.width = 512
    canvas.height = 512

    const widget = {
        type: "GLSL",
        name: "JOVIBALL",
        y: 0,
        inputEl: canvas,
        GL: canvas.getContext('webgl2'),
        FRAGMENT: inputData?.default?.fragment || FRAGMENT_DEFAULT,
        PROGRAM: undefined,
        WIDTH: 512,
        HEIGHT: 512,
        minWidth: 512,
        minHeight: 512,
    }

    widget.setupPositionBuffer = function() {
        const positionBuffer = this.GL.createBuffer();
        this.GL.bindBuffer(this.GL.ARRAY_BUFFER, positionBuffer);
        const positions = [
            -1, -1,
            -1, 1,
            1, -1,
            1, -1,
            -1, 1,
            1, 1,
        ];
        this.GL.bufferData(this.GL.ARRAY_BUFFER, new Float32Array(positions), this.GL.STATIC_DRAW);
        this.GL.bindBuffer(this.GL.ARRAY_BUFFER, positionBuffer);
        const positionAttr = this.GL.getAttribLocation(this.PROGRAM, 'vertexPosition');
        this.GL.vertexAttribPointer(positionAttr, 2, this.GL.FLOAT, false, 0, 0);
        this.GL.enableVertexAttribArray(positionAttr);
    }

    widget.updateParameters = function (texture, time) {
        const textureSamplerLocation = this.GL.getUniformLocation(this.PROGRAM, "textureSampler");
        this.GL.uniform1i(textureSamplerLocation, texture);
        const timeLocation = this.GL.getUniformLocation(this.PROGRAM, "time");
        this.GL.uniform1f(timeLocation, time);
    };

    widget.render = function() {
        if (this.PROGRAM === undefined && this.FRAGMENT != undefined) {
            //console.info(this.FRAGMENT)
            this.initShaderProgram()
            this.GL.useProgram(this.PROGRAM)
            this.setupPositionBuffer()
        }
        this.GL.drawArrays(this.GL.TRIANGLES, 0, 6);
    }

    widget.draw = function(ctx, node, widget_width, y, widget_height) {
        // assign the required style when we are drawn
        Object.assign(this.inputEl.style, get_position_style(ctx, widget_width, y, node.size[1]));
    }

    widget.mouse = function (e, pos, node) {
        if (e.type === 'pointermove') {
            console.info(e.delta);
        }
    }

    widget.computeSize = function (width) {
        return [width, LiteGraph.NODE_WIDGET_HEIGHT]
    }

    widget.initShaderProgram = function() {
        const vertex = this.compileShader(VERTEX_SHADER, this.GL.VERTEX_SHADER);

        const fragment_full = FRAGMENT_HEADER(this.FRAGMENT)
        const fragment = this.compileShader(fragment_full, this.GL.FRAGMENT_SHADER);

        this.PROGRAM = this.GL.createProgram();
        this.GL.attachShader(this.PROGRAM, vertex);
        this.GL.attachShader(this.PROGRAM, fragment);
        this.GL.linkProgram(this.PROGRAM);

        if (!this.GL.getProgramParameter(this.PROGRAM, this.GL.LINK_STATUS)) {
            console.error('Unable to initialize the shader program: ' + this.GL.getProgramInfoLog(this.PROGRAM));
            return;
        }

        // console.info(this.GL.getShaderInfoLog(vertex));
        // console.info(this.GL.getShaderInfoLog(fragment));
        // console.info(this.GL.getProgramInfoLog(this.PROGRAM));
        // console.info('SHADER LINKED');
    }

    widget.compileShader = function(source, type) {
        const shader = this.GL.createShader(type);
        this.GL.shaderSource(shader, source);
        this.GL.compileShader(shader);

        if (!this.GL.getShaderParameter(shader, this.GL.COMPILE_STATUS)) {
            console.error('Shader compilation error: ' + this.GL.getShaderInfoLog(shader));
            this.GL.deleteShader(shader);
            return null;
        }
        return shader;
    }

    document.body.appendChild(widget.inputEl);
    return widget
};

const glsl_node = {
	name: _id + '.js',
    async getCustomWidgets(app) {
        return {
            GLSL: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(GLSLWidget(app, inputData)),
            }),
        }
    },
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === _id) {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const me = onNodeCreated?.apply(this)
                const widget_glsl = this.addCustomWidget(GLSLWidget(app, nodeData))
                this.addCustomWidget(widget_glsl);
                widget_glsl.render();

                let time = 0;

                const onExecutionStart = nodeType.prototype.onExecutionStart;
                nodeType.prototype.onExecutionStart = function (message) {
                    onExecutionStart?.apply(this, arguments);
                    // widget_glsl.updateParameters(texture, time)
                    const timeLocation = widget_glsl.GL.getUniformLocation(widget_glsl.PROGRAM, "time");
                    widget_glsl.GL.uniform1f(timeLocation, time);
                    time += 0.001;

                    widget_glsl.render()
                    this.setOutputData('🖼️', 0)
                    this.setOutputData('😷', 0)
                }

                const onRemoved = nodeType.prototype.onRemoved;
                nodeType.prototype.onRemoved = function (message) {
                    onRemoved?.apply(this, arguments);
                    widget_glsl.inputEl.remove();
                    util.cleanupNode(this);
                };
                return me;
            };
        }
	},

}

app.registerExtension(glsl_node)

