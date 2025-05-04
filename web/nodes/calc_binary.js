/**/

import { app } from "../../../scripts/app.js"
import { widgetHookAB } from "../util.js"

const _id = "OP BINARY (JOV) 🌟"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            widgetHookAB(this, '❓');
            return me;
        }

       return nodeType;
	}
})
