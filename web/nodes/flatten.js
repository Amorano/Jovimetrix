/**
 * File: flatten.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { node_add_dynamic} from '../util/util.js'
import{ hook_widget_size_mode2 } from '../util/util_jov.js'

const _id = "FLATTEN (JOV) ⬇️"
const _prefix = '👾'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        nodeType = node_add_dynamic(nodeType, _prefix);
        nodeType = hook_widget_size_mode2(nodeType);
	}
})
