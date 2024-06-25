/**
 * File: value.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { hook_widget_AB } from '../util/util_jov.js'
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_hide, process_any, widget_type_name } from '../util/util_widget.js'

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
            let track_yyzw = {0:1, 1:1, 2:1, 3:1};

            const widget_rng = this.widgets.find(w => w.name === 'seed');
            const widget_str = this.widgets.find(w => w.name === '📝');

            this.outputs[1].type = "*";
            this.outputs[2].type = "*";
            this.outputs[3].type = "*";
            this.outputs[4].type = "*";

            widget_str.options.menu = false;
            widget_str.origComputeSize = widget_str.computeSize;

            const ab_data = hook_widget_AB(this, '❓');
            const old_callback = ab_data.combo.callback;
            ab_data.combo.callback = () => {
                old_callback?.apply(this);
                widget_hide(this, widget_str, "-jovi");
                widget_str.inputEl.className = "jov-hidden";
                widget_str.computeSize = () => [0, -4];

                this.outputs[0].name = widget_type_name(ab_data.combo.value);
                this.outputs[0].type = ab_data.combo.value;
                let type = ab_data.combo.value;
                if (["LIST", "DICT", "STRING"].includes(ab_data.combo.value)) {
                    process_any(widget_str, ab_data.combo.value);
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                } else {
                    type = "FLOAT";
                    if (ab_data.combo.value.endsWith("INT")) {
                        type = "INT";
                    }
                }
                this.outputs[1].type = type;
                this.outputs[2].type = type;
                this.outputs[3].type = type;
                this.outputs[4].type = type;
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

            setTimeout(() => {
                ab_data.combo.callback();
            }, 10);
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
            if (slotType === TypeSlot.Input) {
                const widget_combo = this.widgets.find(w => w.name === '❓');
                setTimeout(() => { widget_combo.callback(); }, 10);
            }
            return onConnectionsChange?.apply(this, arguments);
        }

    }
})
