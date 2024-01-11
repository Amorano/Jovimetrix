/**
 * File: delay.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import * as util from '../core/util.js'

const _id = "DELAY (JOV) ✋🏽"
let total_timeout = 0;

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            async function python_delay_user(event) {
                const timeout = event.detail?.timeout;
                try {
                    const value = await util.showModal(`
                        <div class="jov-modal-content">
                            <h3 id="jov-delay-header">DELAY NODE #${event.detail?.title || event.detail.id}</h3>
                            <h4>CANCEL OR CONTINUE RENDER?</h4>
                            <div>
                                <button id="jov-submit-cancel">CANCEL</button>
                                <button id="jov-submit-continue">CONTINUE</button>
                            </div>
                        </div>`,
                    (id) => {
                        console.info(id)
                        if (id === "jov-submit-cancel") {
                            return true;
                        } else if (id === "jov-submit-continue") {
                            return false;
                        }
                    }, timeout);

                    total_timeout = 0;
                    var data = { id: event.detail.id, cancel: value }
                    util.api_post('/jovimetrix/message', data);
                }
                catch(e)
                {
                    if (e.message == "TIMEOUT") {
                        total_timeout += timeout;
                        console.info(total_timeout);
                    } else {
                        console.error(e);
                    }
                }
            };
            api.addEventListener("jovi-delay-user", python_delay_user);
            return me;
        }
    }
}
app.registerExtension(ext)
