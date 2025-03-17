/**/

import { app } from "../../../scripts/app.js"
import { widgetHookControl, widgetHookAB } from '../util/util_jov.js'

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
            const alpha = this.widgets.find(w => w.name == '🛟');
            widgetHookControl(this, '❓', alpha, true);
            widgetHookAB(this, '❓', false);
            return me;
        }
        return nodeType;
	}
})
