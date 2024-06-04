/**
 * File: value.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_hide, process_any, process_value, widget_type_name, show_vector, show_boolean } from '../util/util_widget.js'

const _id = "VALUE (JOV) 🧬"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const widget_str = this.widgets.find(w => w.name === '📝');
            const widget_x = this.widgets.find(w => w.name === '🇽');
            const widget_xyzw = this.widgets.find(w => w.name === '🅰️4');
            widget_str.origComputeSize = widget_str.computeSize;
            const combo = this.widgets.find(w => w.name === '❓');
            combo.callback = () => {
                widget_str.inputEl.className = "jov-hidden";
                widget_str.computeSize = () => [0, -4];
                widget_x.options.menu = false;
                widget_xyzw.options.menu = false;
                widget_str.options.menu = false;
                //
                widget_hide(this, widget_x, "-jovi");
                widget_hide(this, widget_xyzw, "-jovi");
                widget_hide(this, widget_str, "-jovi");
                //
                if (combo.value == "BOOLEAN") {
                    show_boolean(widget_x);
                } else if (combo.value == "LIST") {
                    process_any(widget_x, "LIST")
                } else if (combo.value == "DICT") {
                    process_any(widget_x, "DICT")
                } else if (combo.value == "ANY") {
                    process_any(widget_x, "*")
                } else if (combo.value == "MASK") {
                    process_any(widget_x, "MASK")
                } else if (combo.value == "STRING") {
                    process_any(widget_str, "STRING")
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                } else if (combo.value == "FLOAT") {
                    process_value(widget_x, 3);
                } else if (combo.value == "INT") {
                    process_value(widget_x);
                } else if (["VEC2", "COORD2D", "VEC3", "VEC4"].includes(combo.value)) {
                    show_vector(widget_xyzw, 3);
                } else if (["VEC2INT", "VEC3INT", "VEC4INT"].includes(combo.value)) {
                    show_vector(widget_xyzw);
                }
                this.outputs[0].name = widget_type_name(combo.value);
                fitHeight(this);
            }
            setTimeout(() => { combo.callback(); }, 10);
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
            if (slotType === TypeSlot.Input) {
                const combo = this.widgets.find(w => w.name === '❓');
                setTimeout(() => { combo.callback(); }, 10);
            }
            return onConnectionsChange?.apply(this, arguments);
        }
        return nodeType;
	}
})
