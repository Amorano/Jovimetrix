/**
 * File: blend.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import{ widgetSizeModeHook2 } from '../util/util_jov.js'

const _id = "BLEND (JOV) ⚗️"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        widgetSizeModeHook2(nodeType);
	}
})
