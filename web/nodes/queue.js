/**
 * File: queue.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js"
import * as util from '../core/util.js'

const _id = "QUEUE (JOV) 🗃"

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        function update_report(self) {
            self.widget_report.value = `[${self.data_index} / ${self.data_all.length}] ${self.data_current}`;
            app.canvas.setDirty(true);
        }

        function update_list(self) {
            const value = self.widget_queue.value.split('\n');
            self.data_count = value.length;
            self.data_index = 1;
            self.data_current = "";
            update_report(self);
            util.api_cmd_jovian(self.id, "reset");
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;

            let output_data;
            self.data_index = 1;
            self.data_current = "";
            self.data_all = [];

            self.widget_queue = this.widgets[0];
            self.widget_queue.inputEl.addEventListener('input', function (event) {
                update_list(self);
            });

            output_data = this.outputs[0];
            const widget_reset = this.widgets[4];
            const old_callback = widget_reset?.callback;
            widget_reset.callback = async (e) => {
                widget_reset.value = false;
                if (old_callback) {
                    old_callback(this, arguments);
                }
                util.api_cmd_jovian(self.id, "reset");
            }

            self.widget_report = ComfyWidgets.STRING(this, 'QUEUE IS EMPTY 🔜', [
                'STRING', {
                    multiline: true,
                },
            ], app).widget;
            self.widget_report.inputEl.readOnly = true;
            self.widget_report.serializeValue = async () => { };

            async function python_queue_ping(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                self.data_index = event.detail.i;
                self.data_all  = event.detail.l;
                self.data_current = event.detail.c;
                update_report(self);
            }

            // Add names to list control that collapses. And counter to see where we are in the overall
            async function python_queue_done(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                let centerX = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
                let centerY = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
                // util_fun.bewm(centerX / 2, centerY / 3);
                await util.flashBackgroundColor(self.widget_queue.inputEl, 650, 4,  "#995242CC");
            }

            api.addEventListener("jovi-queue-ping", python_queue_ping);
            api.addEventListener("jovi-queue-done", python_queue_done);
            return me;
        }

        const onConnectOutput = nodeType.prototype.onConnectOutput;
        nodeType.prototype.onConnectOutput = function(outputIndex, inputType, inputSlot, inputNode, inputIndex) {
            if (outputIndex == 0) {
                if (inputType == "COMBO") {
                    // can link the "same" list -- user breaks it past that, their problem atm.
                    const widget = inputNode.widgets.find(w => w.name === inputSlot.name);
                    if (this.outputs[0].name != '*' && this.widget_queue.value != widget.options.values.join('\n')) {
                        return false;
                    }
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
                    if (target === undefined) {
                        return;
                    }

                    const widget = node.widgets?.find(w => w.name === target.name);
                    if (widget === undefined) {
                        return;
                    }
                    if (this.outputs[0].name == '*') {
                        this.outputs[0].name = widget.name;
                    }
                    // console.info(widget);
                    if (widget?.origType == "combo" || widget.type == "COMBO") {
                        const values = widget.options.values;
                        // remove all connections that don't match the list?
                        this.widget_queue.value = values.join('\n');
                        update_list(this);
                    }
                } else {
                    this.outputs[0].name = '*';
                }
            }
            return onConnectionsChange?.apply(this, arguments);
        };


    }
}
app.registerExtension(ext)
