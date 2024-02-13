/**
 * File: valuegraph.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { node_add_dynamic } from '../core/util.js'

const _id = "VALUE GRAPH (JOV) 📈"
const _prefix = '❔'

const ext = {
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        nodeType = node_add_dynamic(nodeType, _prefix);
	}
}

app.registerExtension(ext)
