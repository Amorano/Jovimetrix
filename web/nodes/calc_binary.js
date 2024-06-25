/**
 * File: calc_binary.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { TypeSlot } from '../util/util.js'
import { hook_widget_AB } from '../util/util_jov.js'

const _id = "OP BINARY (JOV) 🌟"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const ab_data = hook_widget_AB(this, '❓');
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
            if (slotType === TypeSlot.Input) {
                const widget_combo = this.widgets.find(w => w.name === '❓');
                setTimeout(() => { widget_combo.callback(); }, 10);
            }
            return onConnectionsChange?.apply(this, arguments);
        }
       return nodeType;
	}
})
