/**
 * File: delay.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import * as util from '../core/util.js'
import * as util_fun from '../core/util_fun.js'

const _id = "DELAY (JOV) ✋🏽"

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_timeout = this.widgets[0];
            const widget_wait = this.widgets[1];
            const self = this;
            this.total_timeout = 0;

            async function python_delay_user(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                const timeout = event.detail.timeout;
                try {
                    if (widget_wait.value) {
                        util_fun.bubbles();
                    }
                    const value = await util.showModal(`
                        <div class="jov-modal-content">
                            <h3 id="jov-delay-header">DELAY NODE #${event.detail?.title || event.detail.id} (${self.total_timeout})</h3>
                            <h4>CANCEL OR CONTINUE RENDER?</h4>
                            <div>
                                <button id="jov-submit-continue">CONTINUE</button>
                                <button id="jov-submit-cancel">CANCEL</button>
                            </div>
                        </div>`,
                        (button) => {
                            if (button === "jov-submit-cancel") {
                                return true;
                            } else if (button === "jov-submit-continue") {
                                return false;
                            }
                        }, timeout);

                    window.bubbles_alive = false;
                    var data = { id: event.detail.id, cancel: value };
                    util.api_post('/jovimetrix/message', data);
                } catch (e) {
                    if (e.message !== "TIMEOUT") {
                        console.error(e);
                    } else {
                        self.total_timeout += timeout;
                        app.graph.setDirtyCanvas(true);
                    }
                }
            }
            api.addEventListener("jovi-delay-user", python_delay_user);
            return me;
        }

        const onExecutionStart = nodeType.prototype.onExecutionStart
        nodeType.prototype.onExecutionStart = function (message) {
            onExecutionStart?.apply(this, arguments);
            this.total_timeout = 0;
        }

    }
}
app.registerExtension(ext)
