/**/

import { app } from "../../../scripts/app.js"
import { widgetHookControl, widgetHookValue } from "../util.js"

const _id = "LERP (JOV) 🔰"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            const alpha = this.widgets.find(w => w.name == 'ALPHA');
            widgetHookControl(this, 'TYPE', alpha, true);
            widgetHookValue(this, 'TYPE', 'AA');
            widgetHookValue(this, 'TYPE', 'BB');
            return me;
        }
        return nodeType;
	}
})
