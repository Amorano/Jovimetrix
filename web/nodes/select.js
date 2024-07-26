/**
 * File: select.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeAddDynamic } from '../util/util_node.js'
import { apiJovimetrix } from '../util/util_api.js'

const _id = "SELECT (JOV) 🤏🏽"
const _prefix = '❔'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        nodeType = nodeAddDynamic(nodeType, _prefix);

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const widget_reset = this.widgets.find(w => w.name === 'RESET');
            widget_reset.callback = async() => {
                widget_reset.value = false;
                apiJovimetrix(self.id, "reset");
            }
            return me
        }
	}
})
