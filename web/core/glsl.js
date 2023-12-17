/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js";

const WIDTH = 512;
const HEIGHT = 512;

let GL;
let CANVAS;
let PROGRAM;

const CANVAS_TEMP = document.createElement('canvas');

const vertexTestShader = `
    attribute vec4 a_position;
    void main() {
        gl_Position = a_position;
    }
`;

const fragmentTestShader = `
    precision mediump float;
    void main() {
        gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0); // Blue color
    }
`;

function createShader(type, source) {
    const shader = GL.createShader(type);

    if (!shader) {
        console.error('Unable to create shader of type ' + type);
        return null;
    }
    GL.shaderSource(shader, source);
    GL.compileShader(shader);

    if (!GL.getShaderParameter(shader, GL.COMPILE_STATUS)) {
        console.error('Shader compilation error: ' + GL.getShaderInfoLog(shader));
        GL.deleteShader(shader);
        return null;
    }
    //console.info('Shader compiled successfully');
    return shader;
}

function createProgram(vertex, fragment) {
    const vertexShader = createShader(GL.VERTEX_SHADER, vertex);
    const fragmentShader = createShader(GL.FRAGMENT_SHADER, fragment);

    const program = GL.createProgram();
    GL.attachShader(program, vertexShader);
    GL.attachShader(program, fragmentShader);
    GL.linkProgram(program);

    if (!GL.getProgramParameter(program, GL.LINK_STATUS)) {
        console.error('Unable to initialize the shader program: ' + GL.getProgramInfoLog(program));
        return null;
    }
    //console.info('Shader program linked successfully');
    return program;
}

export async function render(program) {
    const positionBuffer = GL.createBuffer();
    GL.bindBuffer(GL.ARRAY_BUFFER, positionBuffer);
    const positions = [
        -1, -1,
        -1, 1,
        1, -1,
        1, -1,
        -1, 1,
        1, 1,
    ];
    GL.bufferData(GL.ARRAY_BUFFER, new Float32Array(positions), GL.STATIC_DRAW);
    GL.bindBuffer(GL.ARRAY_BUFFER, positionBuffer);
    const positionAttribLocation = GL.getAttribLocation(program, 'a_position');
    GL.vertexAttribPointer(positionAttribLocation, 2, GL.FLOAT, false, 0, 0);
    GL.enableVertexAttribArray(positionAttribLocation);
    GL.useProgram(program);
    GL.drawArrays(GL.TRIANGLES, 0, 6);

    const image = await CANVAS.transferToImageBitmap();

    CANVAS_TEMP.width = image.width;
    CANVAS_TEMP.height = image.height;
    const tempContext = CANVAS_TEMP.getContext('2d');
    tempContext.drawImage(image, 0, 0);
    const dataURL = CANVAS_TEMP.toDataURL('image/png');

    /*
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'rendered_image.png';
    link.click();
    */

    const error = GL.getError();
    if (error !== GL.NO_ERROR) {
        console.error('WebGL error: ' + error);
    }
}

const _id = "GLSL (JOV) 🍩"
const _idjs = _id + ".js";
//const _category = "JOVIMETRIX 🔺🟩🔵/CREATE";
/*
DESCRIPTION = ""
OUTPUT_IS_LIST = (False, False, )
POST = True
*/
const ext = {
	name: _idjs,
    // category: _category,
	async init(app) {
        // Any initial setup to run as soon as the page loads

	},
	async setup(app) {
        // Any setup to run after the app is created
        CANVAS = new OffscreenCanvas(WIDTH, HEIGHT);

        GL = CANVAS.getContext('webgl');
        if (GL === undefined) {
            console.error('Unable to initialize WebGL. Your browser may not support it.');
            return;
        }
        PROGRAM = createProgram(vertexTestShader, fragmentTestShader);
        await render(PROGRAM);
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
        if (nodeData.name === _id) {
            function hookFunctions(proto) {
                if (proto === null) {
                    return;
                }

                const data = Object.getOwnPropertyNames(proto);
                //console.info(data);
                data.forEach(key => {
                    const value = proto[key];
                    if (typeof value === 'function') {
                        proto[key] = function () {
                            // console.info(`Function: ${key}`);
                            return value.apply(this, arguments);
                        };
                    }
                });

                hookFunctions(Object.getPrototypeOf(proto));
            }
            hookFunctions(nodeType.prototype)
        }
/*
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const me = onNodeCreated?.apply(this);
                console.info("MADE SHADER NODE");
                return me
            }

            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, message)
                console.info("RAN SHADER NODE");
                //render(PROGRAM);
            }

            const onAfterExecuteNode = nodeType.prototype.onAfterExecuteNode
            nodeType.prototype.onExecuted = function(param, options) {
                onAfterExecuteNode?.apply(this, param, options)
                console.info("RAN SHADER");
                //render(PROGRAM);
            }

            const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function () {
				onConfigure?.apply(this, arguments);
                console.info("hello")
			};
            console.info(nodeType.prototype)
        }
        */
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
	}
};

app.registerExtension(ext);

