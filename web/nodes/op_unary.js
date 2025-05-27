/**/

import { app } from "../../../scripts/app.js"
import { widgetHookControl } from "../util.js"

const _id = "OP UNARY (JOV) 🎲"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {

        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            const me = await onNodeCreated?.apply(this, arguments);
            await widgetHookControl(this, 'type', 'aa');
            return me;
        }
        return nodeType;
	}
})