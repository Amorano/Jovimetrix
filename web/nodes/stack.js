/**/

import { app } from "../../../scripts/app.js"
import { nodeAddDynamic}  from "../util.js"

const _id = "STACK (JOV) ➕"
const _prefix = 'image'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        await nodeAddDynamic(nodeType, _prefix);
	}
})
