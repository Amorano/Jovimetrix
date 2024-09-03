/**
 * File: flatten.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeAddDynamic } from '../util/util_node.js'
import{ widgetSizeModeHook } from '../util/util_jov.js'

const _id = "FLATTEN (JOV) ⬇️"
const _prefix = '👾'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        widgetSizeModeHook(nodeType);
        nodeAddDynamic(nodeType, _prefix);

	}
})
