/**
 * File: blend.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import{ hook_widget_size_mode } from '../core/util_jov.js'

const _id = "BLEND (JOV) ⚗️"

const ext = {
	name: 'jovimetrix.node.blend',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            hook_widget_size_mode(this);
            return me;
        }
	}
}

app.registerExtension(ext)
