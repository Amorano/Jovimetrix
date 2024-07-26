/**
 * File: tick.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js"
import { api_cmd_jovian } from '../util/util_api.js'

const _id = "TICK (JOV) ⏱";
const EVENT_JOVI_TICK = "jovi-tick";

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const widget_reset = this.widgets.find(w => w.name === 'RESET');
            widget_reset.callback = async() => {
                widget_reset.value = false;
                api_cmd_jovian(self.id, "reset");
            }

            self.widget_count = this.widgets.find(w => w.name === 'VAL');
            async function python_tick(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                self.widget_count.value = event.detail.i;
            }

            api.addEventListener("jovi-tick", python_tick);
            this.onDestroy = () => {
                api.removeEventListener(EVENT_JOVI_TICK, python_tick);
            };
            return me;
        }
	}
})
