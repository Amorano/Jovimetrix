/**
 * File: spout_writer.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { hook_widget_size_mode } from '../util/util_jov.js'

const _id = "SPOUT WRITER (JOV) 🎥"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            hook_widget_size_mode(this);
            return me;
        }
	}
})
