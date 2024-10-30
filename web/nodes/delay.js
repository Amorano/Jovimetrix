/**
 * File: delay.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { domShowModal } from '../util/util.js'
import { apiJovimetrix } from '../util/util_api.js'
import { bubbles } from '../util/util_fun.js'

const _id = "DELAY (JOV) ✋🏽"
const EVENT_JOVI_DELAY = "jovi-delay-user";
const EVENT_JOVI_UPDATE = "jovi-delay-update";

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_time = this.widgets.find(w => w.name == '⏱');
            const widget_enable = this.widgets.find(w => w.name == 'ENABLE');
            this.total_timeout = 0;
            let showing = false;
            let delay_modal;
            const self = this;

            async function python_delay_user(event) {
                if (showing || event.detail.id != self.id) {
                    return;
                }

                if (widget_time.value > 4 && widget_enable.value == true) {
                    bubbles();
                }

                showing = true;
                delay_modal = domShowModal(`
                    <div class="jov-modal-content">
                        <h3 id="jov-delay-header">DELAY NODE #${event.detail?.title || event.detail.id}</h3>
                        <h4>CANCEL OR CONTINUE RENDER?</h4>
                        <div>
                            <button id="jov-submit-continue">CONTINUE</button>
                            <button id="jov-submit-cancel">CANCEL</button>
                        </div>
                    </div>`,
                    (button) => {
                        return (button != "jov-submit-cancel");
                    },
                    widget_time.value);

                let value = false;
                try {
                    value = await delay_modal;
                } catch (e) {
                    if (e.message != "TIMEOUT") {
                        console.error(e);
                    }
                }
                apiJovimetrix(event.detail.id, value);

                showing = false;
                window.bubbles_alive = false;
            }

            async function python_delay_update() {
            }

            api.addEventListener(EVENT_JOVI_DELAY, python_delay_user);
            api.addEventListener(EVENT_JOVI_UPDATE, python_delay_update);

            this.onDestroy = () => {
                api.removeEventListener(EVENT_JOVI_DELAY, python_delay_user);
                api.removeEventListener(EVENT_JOVI_UPDATE, python_delay_update);
            };
            return me;
        }

        const onExecutionStart = nodeType.prototype.onExecutionStart
        nodeType.prototype.onExecutionStart = function() {
            onExecutionStart?.apply(this, arguments);
            self.total_timeout = 0;
        }

    }
})
