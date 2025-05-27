/**/

import { app } from "../../../scripts/app.js"
import { widgetHookControl } from "../util.js"

const _id = "LERP (JOV) 🔰"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            const me = await onNodeCreated?.apply(this, arguments);
            await widgetHookControl(this, 'type', 'alpha', true);
            await widgetHookControl(this, 'type', 'aa');
            await widgetHookControl(this, 'type', 'bb');
            return me;
        }
        return nodeType;
	}
})
