/**/

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { apiJovimetrix } from "../util.js"
import { bubbles } from '../fun.js'

const _id = "DELAY (JOV) ✋🏽"
const EVENT_JOVI_DELAY = "jovi-delay-user";
const EVENT_JOVI_UPDATE = "jovi-delay-update";

function domShowModal(innerHTML, eventCallback, timeout=null) {
    return new Promise((resolve, reject) => {
        const modal = document.createElement("div");
        modal.className = "modal";
        modal.innerHTML = innerHTML;
        document.body.appendChild(modal);

        // center
        const modalContent = modal.querySelector(".jov-modal-content");
        modalContent.style.position = "absolute";
        modalContent.style.left = "50%";
        modalContent.style.top = "50%";
        modalContent.style.transform = "translate(-50%, -50%)";

        let timeoutId;

        const handleEvent = (event) => {
            const targetId = event.target.id;
            const result = eventCallback(targetId);

            if (result != null) {
                if (timeoutId) {
                    clearTimeout(timeoutId);
                    timeoutId = null;
                }
                modal.remove();
                resolve(result);
            }
        };
        modalContent.addEventListener("click", handleEvent);
        modalContent.addEventListener("dblclick", handleEvent);

        if (timeout) {
            timeout *= 1000;
            timeoutId = setTimeout(() => {
                modal.remove();
                reject(new Error("TIMEOUT"));
            }, timeout);
        }

        //setTimeout(() => {
        //    modal.dispatchEvent(new Event('tick'));
        //}, 1000);
    });
}

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = await onNodeCreated?.apply(this, arguments);
            const widget_time = this.widgets.find(w => w.name == 'timer');
            const widget_enable = this.widgets.find(w => w.name == 'enable');
            this.total_timeout = 0;
            let showing = false;
            let delay_modal;
            const self = this;

            async function python_delay_user(event) {
                if (showing || event.detail.id != self.id) {
                    return;
                }

                const time = event.detail.timeout;
                if (time > 4 && widget_enable.value == true) {
                    bubbles();
                    console.info(time, widget_enable.value);
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
                    time);

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
