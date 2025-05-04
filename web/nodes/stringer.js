/**/

import { app } from "../../../scripts/app.js"
import { nodeAddDynamic } from "../util.js"

const _id = "STRINGER (JOV) 🪀"
const _prefix = 'STRING'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        nodeAddDynamic(nodeType, _prefix);
	}
})