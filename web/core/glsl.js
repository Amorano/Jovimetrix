/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"

const CANVAS_TEMP = document.createElement('canvas')

const vertexTestShader = `
    attribute vec4 aVertexPosition;
    void main() {
        // Pass through each vertex position without transforming:
        gl_Position = aVertexPosition;
    }
`

const fragmentTestShader = `
    precision mediump float;
    // Require resolution (canvas size) as an input
    uniform vec3 uResolution;

    void main() {
        // Calculate relative coordinates (uv)
        vec2 uv = gl_FragCoord.xy / uResolution.xy;
        gl_FragColor = vec4(uv.x, uv.y, 0., 1.0);
    }
`

function createShader(GL, type, source) {
    const shader = GL.createShader(type)
    const which = type === GL.VERTEX_SHADER ? 'vertex' : 'fragment'
    if (!shader) {
        console.error('Unable to create shader of type ' + twhichype)
        return null
    }
    GL.shaderSource(shader, source)
    GL.compileShader(shader)

    if (!GL.getShaderParameter(shader, GL.COMPILE_STATUS)) {
        console.error('Shader compilation error: ' + GL.getShaderInfoLog(shader))
        GL.deleteShader(shader)
        return null
    }
    console.info(which + ' shader compiled successfully')
    return shader
}

function createProgram(GL, vertex, fragment) {
    const vertexShader = createShader(GL, GL.VERTEX_SHADER, vertex)
    if (vertexShader === null){
        return
    }
    const fragmentShader = createShader(GL, GL.FRAGMENT_SHADER, fragment)
    if (fragmentShader === null){
        return
    }

    const program = GL.createProgram()
    GL.attachShader(program, vertexShader)
    GL.attachShader(program, fragmentShader)
    GL.linkProgram(program)

    if (!GL.getProgramParameter(program, GL.LINK_STATUS)) {
        console.error('Unable to initialize the shader program: ' + GL.getProgramInfoLog(program))
        return null
    }
    // gl.deleteShader(vertexShader);
    // gl.deleteShader(fragmentShader);
    console.info('Shader program linked successfully')
    return program
}

export async function render(GL, CANVAS, program) {
    const positionBuffer = GL.createBuffer()
    GL.bindBuffer(GL.ARRAY_BUFFER, positionBuffer)
    const positions = [
        -1, -1,
        -1, 1,
        1, -1,
        1, -1,
        -1, 1,
        1, 1,
    ]
    GL.bufferData(GL.ARRAY_BUFFER, new Float32Array(positions), GL.STATIC_DRAW)
    GL.bindBuffer(GL.ARRAY_BUFFER, positionBuffer)
    const positionAttribLocation = GL.getAttribLocation(program, 'aVertexPosition')
    GL.vertexAttribPointer(positionAttribLocation, 2, GL.FLOAT, false, 0, 0)
    GL.enableVertexAttribArray(positionAttribLocation)
    GL.useProgram(program)
    GL.drawArrays(GL.TRIANGLES, 0, 6)

    const image = await CANVAS.transferToImageBitmap()

    CANVAS_TEMP.width = image.width
    CANVAS_TEMP.height = image.height
    const tempContext = CANVAS_TEMP.getContext('2d')
    tempContext.drawImage(image, 0, 0)
    const dataURL = CANVAS_TEMP.toDataURL('image/png')
    /*
    const link = document.createElement('a')
    link.href = dataURL
    link.download = 'rendered_image.png'
    link.click()*/

    const error = GL.getError()
    if (error !== GL.NO_ERROR) {
        console.error('WebGL error: ' + error)
    }
}

const _id = "GLSL (JOV) 🍩"
const ext = {
	name: _id + '.js',
    // category: _category,
	async init(app) {
        // Any initial setup to run as soon as the page loads

	},
	async setup(app) {

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
            //console.info(nodeData.name);
            let GL = null;
            let VERTEX = null;
            let FRAGMENT = null;
            let PROGRAM = null;
            let WIDTH = 512
            let HEIGHT = 512
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const me = onNodeCreated?.apply(this)
                // console.info(this)
                let CANVAS = new OffscreenCanvas(WIDTH, HEIGHT)
                GL = CANVAS.getContext('webgl2')
                if (GL === undefined) {
                    console.error('Unable to initialize WebGL. Your browser may not support it.')
                } else {
                    console.info("MADE SHADER NODE")
                    PROGRAM = createProgram(GL, vertexTestShader, fragmentTestShader)
                    render(GL, CANVAS, PROGRAM)
                }
                this.onRemoved = function () {
                    // util.cleanupNode(this);

                };
                return me
            }

            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function(message) {
                const me = onExecuted?.apply(this, message)
                render(GL, CANVAS, PROGRAM)
                console.info("RAN SHADER NODE")
                return me
            }
        }
    /*

            const onAfterExecuteNode = nodeType.prototype.onAfterExecuteNode
            nodeType.prototype.onExecuted = function(param, options) {
                onAfterExecuteNode?.apply(this, param, options)
                console.info("RAN SHADER")
                //render(PROGRAM)
            }

            const onConfigure = nodeType.prototype.onConfigure
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments)
                console.info("hello")
            }
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
}

app.registerExtension(ext)

