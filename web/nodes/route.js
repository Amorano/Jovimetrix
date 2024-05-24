/**
 * File: route.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { node_add_dynamic2 } from '../util/util.js'

const _id = "ROUTE (JOV) 🚌"
const _prefix = '🔮'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        nodeType = node_add_dynamic2(nodeType, _prefix, '*', 1);
	}
})
