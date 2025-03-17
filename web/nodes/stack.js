/**/

import { app } from "../../../scripts/app.js"
import { nodeAddDynamic} from '../util/util_node.js'

const _id = "STACK (JOV) ➕"
const _prefix = '👾'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }
        nodeAddDynamic(nodeType, _prefix);
	}
})
