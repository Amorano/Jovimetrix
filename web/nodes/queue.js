/**/

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from '../../../scripts/widgets.js';
import { apiJovimetrix, TypeSlotEvent, TypeSlot } from "../util.js"
import { flashBackgroundColor } from '../fun.js'

const _id1 = "QUEUE (JOV) 🗃";
const _id2 = "QUEUE TOO (JOV) 🗃";
const _prefix = '❔';
const EVENT_JOVI_PING = "jovi-queue-ping";
const EVENT_JOVI_DONE = "jovi-queue-done";

app.registerExtension({
	name: 'jovimetrix.node.' + _id1,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name != _id1 && nodeData.name != _id2) {
            return;
        }

        function update_report(self) {
            self.widget_report.value = `[${self.data_index+1} / ${self.data_all.length}]\n${self.data_current}`;
            app.canvas.setDirty(true);
        }

        function update_list(self, value) {
            self.data_count = value.length;
            self.data_index = 1;
            self.data_current = "";
            update_report(self);
            apiJovimetrix(self.id, "reset");
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = await onNodeCreated?.apply(this, arguments);
            const self = this;
            this.data_index = 1;
            this.data_current = "";
            this.data_all = [];
            this.widget_report = ComfyWidgets.STRING(this, 'QUEUE IS EMPTY 🔜', [
                'STRING', {
                    multiline: true,
                },
            ], app).widget;
            this.widget_report.inputEl.readOnly = true;
            this.widget_report.serializeValue = async () => { };

            const widget_queue = this.widgets.find(w => w.name == 'queue');
            const widget_batch = this.widgets.find(w => w.name == 'batch');
            const widget_hold = this.widgets.find(w => w.name == 'hold');
            const widget_reset = this.widgets.find(w => w.name == 'reset');

            widget_queue.inputEl.addEventListener('input', function () {
                const value = widget_queue.value.split('\n');
                update_list(self, value);
            });

            widget_reset.callback = () => {
                widget_reset.value = false;
                apiJovimetrix(self.id, "reset");
            }

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
                await flashBackgroundColor(self.widget_queue.inputEl, 650, 4, "#995242CC");
            }

            api.addEventListener(EVENT_JOVI_PING, python_queue_ping);
            api.addEventListener(EVENT_JOVI_DONE, python_queue_done);

            this.onDestroy = () => {
                api.removeEventListener(EVENT_JOVI_PING, python_queue_ping);
                api.removeEventListener(EVENT_JOVI_DONE, python_queue_done);
            };

            setTimeout(() => { widget_hold.callback(); }, 5);
            setTimeout(() => { widget_batch.callback(); }, 5);
            return me;
        }

        const onConnectOutput = nodeType.prototype.onConnectOutput;
        nodeType.prototype.onConnectOutput = function(outputIndex, inputType, inputSlot, inputNode) {
            if (outputIndex == 0 && inputType == "COMBO") {
                // can link the "same" list -- user breaks it past that, their problem atm
                const widget_queue = this.widgets.find(w => w.name == 'queue');
                const widget = inputNode.widgets.find(w => w.name == inputSlot.name);
                widget_queue.value = widget.options.values.join('\n');
            }
            return onConnectOutput?.apply(this, arguments);
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info)
        //side, slot, connected, link_info
        {
            if (slotType == TypeSlot.Output && slot == 0 && link_info && event == TypeSlotEvent.Connect) {
                const node = app.graph.getNodeById(link_info.target_id);
                if (node === undefined || node.inputs === undefined) {
                    return;
                }
                const target = node.inputs[link_info.target_slot];
                if (target === undefined) {
                    return;
                }

                const widget = node.widgets?.find(w => w.name == target.name);
                if (widget === undefined) {
                    return;
                }
                this.outputs[0].name = widget.name;
                if (widget?.origType == "combo" || widget.type == "COMBO") {
                    const values = widget.options.values;
                    const widget_queue = this.widgets.find(w => w.name == 'queue');
                    // remove all connections that don't match the list?
                    widget_queue.value = values.join('\n');
                    update_list(this, values);
                }
                this.outputs[0].name = _prefix;
            }
            return onConnectionsChange?.apply(this, arguments);
        };
    }
})
