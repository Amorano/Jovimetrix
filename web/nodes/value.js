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

            const widget_rng = this.widgets.find(w => w.name === 'seed');
            const widget_str = this.widgets.find(w => w.name === '📝');

            this.outputs[1].type = "*";
            this.outputs[2].type = "*";
            this.outputs[3].type = "*";
            this.outputs[4].type = "*";

            widget_str.options.menu = false;
            widget_str.origComputeSize = widget_str.computeSize;

            const ab_data = hook_widget_AB(this, '❓');
            const oldCallback = ab_data.combo.callback;
            ab_data.combo.callback = () => {
                oldCallback?.apply(this, arguments);
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
            return me;
        }
    }
})
