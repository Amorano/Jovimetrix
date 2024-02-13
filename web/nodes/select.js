/**
 * File: select.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { node_add_dynamic } from '../core/util.js'
import { api_cmd_jovian } from '../core/util_api.js'

const _id = "SELECT (JOV) 🤏🏽"
const _prefix = '❔'

const ext = {
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        nodeType = node_add_dynamic(nodeType, _prefix);

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const widget_reset = this.widgets[1];
            widget_reset.callback = async (e) => {
                widget_reset.value = false;
                api_cmd_jovian(self.id, "reset");
            }
            return me
        }
	}
}

app.registerExtension(ext)
