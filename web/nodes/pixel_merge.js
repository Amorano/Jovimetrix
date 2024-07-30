/**
 * File: pixel_merge.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import{ widgetSizeModeHook } from '../util/util_jov.js'

const _id = "PIXEL MERGE (JOV) 🫂"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            widgetSizeModeHook(this);
            return me;
        }
	}
})
