/**
 * File: queue.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import * as util from '../core/util.js'
import * as fun from '../core/fun.js'

const _id = "QUEUE (JOV) 🗃"

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        let output_data;
        let widget_queue;
        let widget_current;
        let widget_index = 1;
        let widget_count = 0;
        let widget_label;

        function update_list() {
            const data = widget_queue.value.split('\n');
            widget_count = data.length;
            widget_index = 1;
            widget_current = data[0];
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            widget_queue = this.widgets[0];
            widget_queue.inputEl.addEventListener('input', function (event) {
                update_list();
            });

            output_data = this.outputs[0];
            const widget_reset = this.widgets[2];
            const old_callback = widget_reset?.callback;
            widget_reset.callback = async (e) => {
                this.widgets[2].value = false;
                if (old_callback) {
                    old_callback(this, arguments);
                }

                const data = {
                    id: self.id,
                    cmd: "reset"
                };
                util.api_post('/jovimetrix/message', data);
            }

            const onConnectOutput = nodeType.prototype.onConnectOutput;
            nodeType.prototype.onConnectOutput = function(outputIndex, inputType, inputSlot, inputNode, inputIndex) {
                if (outputIndex == 0) {
                    if (inputType != "text" && inputType != "COMBO"){
                        return false;
                    }

                    // can link the "same" list -- user breaks it past that, their problem atm.
                    const widget = inputNode.widgets.find(w => w.name === inputSlot.name);
                    if (this.outputs[0].name != '*' && widget_queue.value != widget.options.values.join('\n')) {
                        return false;
                    }
                }
                return onConnectOutput?.apply(this, arguments);
            }

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (side, slot, connected, link_info)
            {
                if (slot == 0 && side == 2) {
                    if (connected) {
                        const node = app.graph.getNodeById(link_info.target_id);
                        const target = node.inputs[link_info.target_slot];
                        const widget = node.widgets.find(w => w.name === target.name);
                        if (this.outputs[0].name == '*') {
                            this.outputs[0].name = widget.name;
                        }
                        const values = widget.options.values;
                        // remove all connections that don't match the list?
                        widget_queue.value = values.join('\n');
                        update_list();
                    } else {
                        this.outputs[0].name = '*';
                    }
                }
                return onConnectionsChange?.apply(this, arguments);
            };

            async function python_queue_list(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                console.info(event.detail.data);
            }

            async function python_queue_ping(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                // fun.bewm(self.pos[0], self.pos[1]);
            }

            // Add names to list control that collapses. And counter to see where we are in the overall

            async function python_queue_done(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                let centerX = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
                let centerY = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;

                fun.bewm(centerX / 2, centerY / 3);
                await util.flashBackgroundColor(widget_queue.inputEl, 650, 4,  "#995242CC");
            }
            api.addEventListener("jovi-queue-list", python_queue_list);
            api.addEventListener("jovi-queue-ping", python_queue_ping);
            api.addEventListener("jovi-queue-done", python_queue_done);
            return me;
        }
    }
}
app.registerExtension(ext)
