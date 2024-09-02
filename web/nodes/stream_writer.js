/**
 * File: stream_writer.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { widgetSizeModeHook2 } from '../util/util_jov.js'

const _id = "STREAM WRITER (JOV) 🎞️"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        widgetSizeModeHook2(nodeType);
	}
})
