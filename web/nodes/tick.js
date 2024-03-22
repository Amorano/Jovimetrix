/**
 * File: tick.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js"
import { api_cmd_jovian } from '../util/util_api.js'

const _id = "TICK (JOV) ⏱"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const widget_reset = this.widgets.find(w => w.name === 'RESET');
            widget_reset.callback = async (e) => {
                widget_reset.value = false;
                api_cmd_jovian(self.id, "reset");
            }

            self.widget_count = this.widgets.find(w => w.name === '#️⃣');
            async function python_tick(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                self.widget_count.value = event.detail.i;
            }
            api.addEventListener("jovi-tick", python_tick);
            return me;
        }
	}
})
