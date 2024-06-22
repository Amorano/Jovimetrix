/**
 * File: value.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_show, widget_hide, process_any, widget_type_name, show_vector } from '../util/util_widget.js'

const _id = "VALUE (JOV) 🧬"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            const widget_x4 = this.widgets.find(w => w.name === '🅰️🅰️');
            const widget_y4 = this.widgets.find(w => w.name === '🅱️🅱️');
            widget_x4.options.menu = false;
            widget_y4.options.menu = false;
            let bool_x = {0:false}
            let bool_y = {0:false}
            let track_xyzw = {0:0, 1:0, 2:0, 3:0};
            let track_yyzw = {0:0, 1:0, 2:0, 3:0};

            const widget_rng = this.widgets.find(w => w.name === 'seed');
            const widget_str = this.widgets.find(w => w.name === '📝');
            this.outputs[1].type = "*";
            this.outputs[2].type = "*";
            this.outputs[3].type = "*";
            this.outputs[4].type = "*";
            const self = this;

            widget_str.options.menu = false;
            widget_str.origComputeSize = widget_str.computeSize;

            const widget_combo = this.widgets.find(w => w.name === '❓');
            widget_combo.callback = () => {
                function set_output(what) {
                    self.outputs[1].type = what;
                    self.outputs[2].type = what;
                    self.outputs[3].type = what;
                    self.outputs[4].type = what;
                }

                widget_hide(this, widget_x4, "-jovi");
                widget_hide(this, widget_y4, "-jovi");
                widget_hide(this, widget_str, "-jovi");
                //widget_hide(this, widget_rng, "-jovi");
                widget_str.inputEl.className = "jov-hidden";
                widget_str.computeSize = () => [0, -4];

                this.outputs[0].name = widget_type_name(widget_combo.value);
                this.outputs[0].type = widget_combo.value;

                if (["LIST", "DICT", "STRING"].includes(widget_combo.value)) {
                    process_any(widget_str, widget_combo.value);
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                    set_output(widget_combo.value);
                } else {
                    let type = "FLOAT";
                    if (widget_combo.value.endsWith("INT")) {
                        type = "INT";
                    }
                    const data_x = (widget_combo.value === "BOOLEAN") ? bool_x : track_xyzw;
                    const data_y = (widget_combo.value === "BOOLEAN") ? bool_y : track_yyzw;
                    widget_show(widget_x4)
                    show_vector(widget_x4, data_x, widget_combo.value);
                    set_output(type);
                    if (widget_rng.value > 0) {
                        widget_show(widget_y4)
                        show_vector(widget_y4, data_y, widget_combo.value);
                    }
                }
                fitHeight(this);
            }

            widget_x4.callback = () => {
                if (widget_x4.type === "toggle") {
                    bool_x[0] = widget_x4.value;
                } else {
                    Object.keys(widget_x4.value).forEach((key) => {
                        track_xyzw[key] = widget_x4.value[key];
                    });
                }
            }

            widget_y4.callback = () => {
                if (widget_y4.type === "toggle") {
                    bool_y[0] = widget_y4.value;
                } else {
                    Object.keys(widget_y4.value).forEach((key) => {
                        track_yyzw[key] = widget_y4.value[key];
                    });
                }
            }

            widget_rng.callback = () => {
                widget_hide(this, widget_y4, "-jovi");
                if (widget_rng.value > 0 && !["STRING", "DICT", "LIST"].includes(widget_combo.value)) {
                    widget_show(widget_y4)
                    show_vector(widget_y4, widget_combo.value);
                }
                fitHeight(this);
            }

            setTimeout(() => {
                widget_combo.callback();
            }, 10);
            setTimeout(() => {
                widget_rng.callback();
            }, 10);

            return me;
        }

        /*
        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
            if (slotType === TypeSlot.Input) {
                const widget_combo = this.widgets.find(w => w.name === '❓');
                setTimeout(() => { widget_combo.callback(); }, 10);
            }
            return onConnectionsChange?.apply(this, arguments);
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const me = onExecuted?.apply(this,arguments);
            let values = message["text"].toString().map(Number);
            this.outputs[1]["name"] = values[1] + " width"
            this.outputs[2]["name"] = values[2] + " height"
            this.outputs[3]["name"] = values[0] + " count"
            return me;
        }*/
    }
})
