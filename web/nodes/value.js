/**
 * File: value.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_hide, widget_show, widget_type_name } from '../util/util_widget.js'

const _id = "VALUE (JOV) 🧬"

function process_value(input, widget, subtype="FLOAT") {
    widget_show(widget);
    input.type = subtype;
    if (subtype === "BOOLEAN") {
        widget.type = "toggle";
    } else if (subtype === "FLOAT" || subtype === "INT") {
        widget.type = "number";
        if (widget?.options) {
            if (subtype=="FLOAT") {
                widget.options.precision = 3;
                widget.options.step = 1;
                widget.options.round = 0.1;
            } else {
                widget.options.precision = 0;
                widget.options.step = 10;
                widget.options.round = 1;
            }
        }
    } else {
        widget.type = subtype;
    }
}

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
            widget_str.origComputeSize = widget_str.computeSize;
            const combo = this.widgets.find(w => w.name === '❓');
            combo.callback = () => {
                widget_str.inputEl.className = "jov-hidden";
                widget_str.computeSize = () => [0, -4];
                const in_x = this.inputs.find(w => w.name === 'X');
                const in_y = this.inputs.find(w => w.name === 'Y');
                const in_z = this.inputs.find(w => w.name === 'Z');
                const in_w = this.inputs.find(w => w.name === 'W');
                const widget_x = this.widgets.find(w => w.name === '🇽');
                const widget_y = this.widgets.find(w => w.name === '🇾');
                const widget_z = this.widgets.find(w => w.name === '🇿');
                const widget_w = this.widgets.find(w => w.name === '🇼');
                widget_x.options.forceInput = true;
                widget_y.options.forceInput = true;
                widget_z.options.forceInput = true;
                widget_w.options.forceInput = true;
                widget_str.options.forceInput = true;
                widget_hide(this, widget_x, "-jovi");
                widget_hide(this, widget_y, "-jovi");
                widget_hide(this, widget_z, "-jovi");
                widget_hide(this, widget_w, "-jovi");
                widget_hide(this, widget_str, "-jovi");
                if (combo.value == "BOOLEAN") {
                    process_value(in_x, widget_x, "BOOLEAN")
                } else if (combo.value == "LIST") {
                    process_value(in_x, widget_x, "LIST")
                } else if (combo.value == "DICT") {
                    process_value(in_x, widget_x, "DICT")
                } else if (combo.value == "ANY") {
                    process_value(in_x, widget_x, "*")
                } else if (combo.value == "MASK") {
                    process_value(in_x, widget_x, "MASK")
                } else if (combo.value == "STRING") {
                    process_value(in_x, widget_str, "STRING")
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                } else if (combo.value == "FLOAT") {
                    process_value(in_x, widget_x);
                } else if (combo.value == "INT") {
                    process_value(in_x, widget_x, "INT");
                } else if (combo.value == "VEC2" || combo.value == "COORD2D") {
                    process_value(in_x, widget_x);
                    process_value(in_y, widget_y);
                } else if (combo.value == "VEC2INT") {
                    process_value(in_x, widget_x, "INT");
                    process_value(in_y, widget_y, "INT");
                } else if (combo.value == "VEC3") {
                    process_value(in_x, widget_x);
                    process_value(in_y, widget_y);
                    process_value(in_z, widget_z);
                } else if (combo.value == "VEC3INT") {
                    process_value(in_x, widget_x, "INT");
                    process_value(in_y, widget_y, "INT");
                    process_value(in_z, widget_z, "INT");
                } else if (combo.value == "VEC4") {
                    process_value(in_x, widget_x);
                    process_value(in_y, widget_y);
                    process_value(in_z, widget_z);
                    process_value(in_w, widget_w);
                } else if (combo.value == "VEC4INT") {
                    process_value(in_x, widget_x, "INT");
                    process_value(in_y, widget_y, "INT");
                    process_value(in_z, widget_z, "INT");
                    process_value(in_w, widget_w, "INT");
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
