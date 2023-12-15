/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js";

const WIDTH = 512;
const HEIGHT = 512;

function createProgram(gl, vertex, fragment) {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertex);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragment);

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Unable to initialize the shader program: ' + gl.getProgramInfoLog(program));
        return null;
    }
    return program;
}

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compilation error: ' + this.gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function saveImageData(imageData) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.putImageData(imageData, 0, 0);
    const dataURL = tempCanvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'rendered_image.png';
    link.click();
}

function render(gl) {
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);
    gl.drawArrays(gl.POINTS, 0, 1);

    const imageData = offscreenCanvas.getContext('2d').getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
    saveImageData(imageData);
}

const ext = {
	name: "jovimetrix.glsl",
    category: "JOVIMETRIX \ud83d\udd3a\ud83d\udfe9\ud83d\udd35/CREATE",
	async init(app) {
        // Any initial setup to run as soon as the page loads
        this.vertexShader = `
            attribute vec4 a_position;
            void main() {
                gl_Position = a_position;
            }
        `;

        this.fragmentShader = `
            precision mediump float;
            void main() {
                gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red color
            }
        `;
	},
	async setup(app) {
        // Any setup to run after the app is created
        const offscreenCanvas = new OffscreenCanvas(WIDTH, HEIGHT);
        this.gl = offscreenCanvas.getContext('webgl');

        if (this.gl === undefined) {
            console.error('Unable to initialize WebGL. Your browser may not support it.');
            return;
        }

        const program = createProgram(this.gl, this.vertexShader, this.fragmentShader);
        if (!program) {
            return;
        }
	},
	async addCustomNodeDefs(defs, app) {
		// Object.keys(defs)
        // Add custom node definitions
		// These definitions will be configured and registered automatically
		// defs is a lookup core nodes, add yours into this
	},
	async getCustomWidgets(app) {
		// Return custom widget types
		// See ComfyWidgets for widget examples
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Run custom logic before a node definition is registered with the graph
	},
	async registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexibility than a custom node def

	},
	loadedGraphNode(node, app) {
		// Fires for each node when loading/dragging/etc a workflow json or png
		// If you break something in the backend and want to patch workflows in the frontend
		// This is the place to do this
	},
	nodeCreated(node, app) {
		// Fires every time a node is constructed
		// You can modify widgets/add handlers/etc here
        render();
	}
};

app.registerExtension(ext);
