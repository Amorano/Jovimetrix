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
out vec2 iCoord;
void main() {
    gl_Position = vec4(iPosition, 0.0, 1.0);
    iCoord = iPosition * 0.5 + 0.5;
}`

const FRAGMENT_HEADER = (body) => {
    return `#version 300 es
#ifdef GL_ES
precision mediump float;
#endif

#define PI 3.14159265359

precision highp float;
in vec2 iCoord;
uniform vec2 iResolution;
uniform sampler2D iChannel0;
uniform float iTime;
out vec4 FragColor;

` + body};

const FRAGMENT_DEFAULT3 = `
float plot(vec2 st) {
    return smoothstep(0.02, 0.0, abs(st.y - st.x));
}

void main() {
    float y = iCoord.x;
    vec3 color = vec3(y);
    float pct = plot(iCoord);
    color = (1.0 - pct) * color + pct * vec3(0.0, 1.0, 0.0);
    FragColor = vec4(color, 1.0);
}`

const FRAGMENT_DEFAULT2 = `void main() {
    vec4 color = texture(iChannel0, iCoord);
    FragColor = vec4(iCoord.x, iCoord.y, 0.0, 1.0);
}`

const FRAGMENT_DEFAULT4 = `void main() {
	vec3 c;
	float l, z = iTime;
	for(int i=0; i < 3; i++) {
        vec2 uv, p = iCoord;
		p -= .5;
		p.x *= iResolution.x / iResolution.y;
		z += .07;
		l = length(p);
		uv += p / l * (sin(z) + 1.) * abs(sin(l * 9. - z - z));
		c[i] = .01 / length(mod(uv, 1.) - .5);
	}
	FragColor = vec4(c / l, iTime);
}`

const FRAGMENT_DEFAULT12 = `void main() {
    // the sound texture is 512x2
    int tx = int(iCoord.x * 512.0);

	// first row is frequency data (48Khz/4 in 512 texels, meaning 23 Hz per texel)
	float fft  = texelFetch( iChannel0, ivec2(tx,0), 0 ).x;

    // second row is the sound wave, one texel is one mono sample
    float wave = texelFetch( iChannel0, ivec2(tx,1), 0 ).x;

	// convert frequency to colors
	vec3 col = vec3( fft, 4.0*fft*(1.0-fft), 1.0-fft ) * fft;

    // add wave form on top
	col += 1.0 -  smoothstep( 0.0, 0.15, abs(wave - iCoord.y) );

	// output final color
	FragColor = vec4(col,1.0);
}`

const FRAGMENT_DEFAULT = `vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{
    return a + b*cos( 6.28318*(c*t+d) );
}

void main() {
    float aspect = iResolution.y / iResolution.x;
    vec2 center = vec2(0.5, aspect * 0.5);
    float dist = length(iCoord - center) ;
    vec3 music = texture(iChannel0, vec2(dist * 0.1, 0.0)).rgb ;
    float a = smoothstep(0.0, 1.0, pow(length(music), 2.0) );

    vec3 c1 = vec3(255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0);
    vec3 c2 = vec3(255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0);
    vec3 c3 = vec3(255.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0);
    vec3 c4 = vec3(0.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0);

    vec3 col = a * palette(length(music) * 2.0, c1, c4, c2, c2 );
    FragColor = vec4(col,1.0);
}`


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
        vertex_shader: compileShader(VERTEX_SHADER, GL.VERTEX_SHADER),
        initShaderProgram() {
            const fragment_full = FRAGMENT_HEADER(this.FRAGMENT);
            const fragment = compileShader(fragment_full, GL.FRAGMENT_SHADER);

            if (!fragment) {
                console.error(GL.getShaderInfoLog(fragment));
                this.compiled = false;
                return null;
            }

            PROGRAM = GL.createProgram();
            GL.attachShader(PROGRAM, widget.vertex_shader);
            GL.attachShader(PROGRAM, fragment);
            GL.linkProgram(PROGRAM);

            if (!GL.getProgramParameter(PROGRAM, GL.LINK_STATUS)) {
                console.error('Unable to initialize the shader program: ' + GL.getProgramInfoLog(PROGRAM));
                console.error(GL.getShaderInfoLog(fragment));
                console.error(GL.getProgramInfoLog(PROGRAM));
                this.compiled = false;
                return null;
            }

            GL.useProgram(PROGRAM);
            this.compiled = true;

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
        frame() {
            const pixels = new Uint8Array(canvas.width * canvas.height * 4);
            GL.readPixels(0, 0, canvas.width, canvas.height, GL.RGBA, GL.UNSIGNED_BYTE, pixels);
            const img = new Image();
            img.src = canvas.toDataURL();
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
        },
        async serializeValue(nodeId, widgetIndex) {
            if (widgetIndex !== 5) {
                return;
            }
            return this.frame();
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
            this.serialize_widgets = true;
            const widget_time = this.widgets[0];
            const widget_fixed = this.widgets[1];
            const widget_reset = this.widgets[2];
            const widget_wh = this.widgets[3];
            const widget_glsl = this.addCustomWidget(GLSLWidget(app, 'GLSL', FRAGMENT_DEFAULT))
            widget_glsl.render()

            const widget_fragment = this.widgets[4];
            widget_fragment.inputEl.addEventListener('input', function (event) {
                // console.log('Textarea content changed:', event.target.value);
                widget_glsl.FRAGMENT = event.target.value;
                widget_glsl.initShaderProgram();
            });

            let TIME = 0
            const onExecutionStart = nodeType.prototype.onExecutionStart;
            nodeType.prototype.onExecutionStart = function (message) {
                onExecutionStart?.apply(this, arguments);
                if (TIME == 0) {
                    widget_time.value = 0
                }
                if (widget_reset.value == false) {
                    if (widget_fixed.value > 0) {
                        widget_time.value += widget.fixed.value;
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
                widget_glsl.update_resolution(widget_wh.value[0], widget_wh.value[1]);
                widget_glsl.update_time(widget_time.value)
                widget_glsl.render();
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
                widget_glsl.inputEl.remove();
                util.cleanupNode(this);
            };
            return me;
        }
    }
}

app.registerExtension(ext)
