/**
 * File: flatten.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { node_add_dynamic} from '../util/util.js'

const _id = "FLATTEN (JOV) ⬇️"
const _prefix = '👾'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        nodeType = node_add_dynamic(nodeType, _prefix);
	}
})
