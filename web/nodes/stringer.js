/**
 * File: stringer.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeAddDynamic } from '../util/util_node.js'

const _id = "STRINGER (JOV) 🪀"
const _prefix = '❔'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }
        nodeType = nodeAddDynamic(nodeType, _prefix);
	}
})
