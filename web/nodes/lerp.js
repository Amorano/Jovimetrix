/**
 * File: lerp
 * .js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { hook_widget_AB } from '../util/util_jov.js'

const _id = "LERP (JOV) 🔰"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            hook_widget_AB(this, '❓', 0);
            return me;
        }
        return nodeType;
	}
})
