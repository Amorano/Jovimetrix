/**
 * File: streamreader.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"

const _id = "STREAM READER (JOV) 📺"

const ext = {
	name: 'jovimetrix.node.streamreader',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onExecutionStart = nodeType.prototype.onExecutionStart
        nodeType.prototype.onExecutionStart = function (message) {
            onExecutionStart?.apply(this, arguments)

        }
    }
}

app.registerExtension(ext)
