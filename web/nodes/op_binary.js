/**/

import { app } from "../../../scripts/app.js"
import { widgetHookControl } from "../util.js"

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
            console.info(this);
            widgetHookControl(this, 'type', 'aa');
            widgetHookControl(this, 'type', 'bb');
            return me;
        }

       return nodeType;
	}
})
