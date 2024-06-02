/**
 * File: calc_binary.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_hide, widget_show, widget_type_name } from '../util/util_widget.js'

const _id = "OP BINARY (JOV) 🌟"

function process_value(input, widget, precision=0, visible=false, typ="number", subtype="FLOAT") {
    if (input === undefined) {
        if (visible) {
            widget_show(widget);
            widget.origType = widget.type;
            widget.type = typ;
        }
    } else {
        input.origType = input.Type;
        input.type = subtype;
    }
    if (widget?.options) {
        widget.options.precision = precision;
        if (precision == 0) {
            widget.options.step = 10;
            widget.options.round = 1;
        } else {
            widget.options.step = 1;
            widget.options.round =  0.1;
        }
    }
}

function show_boolean(widget_x, in_x, visible_x, typ="toggle") {
    if (!in_x && visible_x) {
        widget_show(widget_x);
        widget_x.origType = widget_x.type;
        widget_x.type = typ;
    }
}

function show_vector(widget, precision=0) {
    widget_show(widget);
    widget.origType = widget.type;
    if (precision == 0) {
        widget.options.step = 1;
        widget.options.round = 1;
        widget.options.precision = 0;
    } else {
        widget.options.step = 1 / (10^Math.max(1, precision-2));
        widget.options.round =  1 / (10^Math.max(1, precision-1));
        widget.options.precision = precision;
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
            const combo = this.widgets.find(w => w.name === '❓');
            combo.callback = () => {
                const in_x = this.inputs.find(w => w.name === '🇽');
                const in_y = this.inputs.find(w => w.name === '🇾');
                //
                const widget_a = this.inputs.find(w => w.name === '🅰️');
                const widget_b = this.inputs.find(w => w.name === '🅱️');
                //
                const widget_x = this.widgets.find(w => w.name === '🇽');
                const widget_xy = this.widgets.find(w => w.name === '🅰️2');
                const widget_xyz = this.widgets.find(w => w.name === '🅰️3');
                const widget_xyzw = this.widgets.find(w => w.name === '🅰️4');
                const widget_y = this.widgets.find(w => w.name === '🇾');
                const widget_yy = this.widgets.find(w => w.name === '🅱️2');
                const widget_yyz = this.widgets.find(w => w.name === '🅱️3');
                const widget_yyzw = this.widgets.find(w => w.name === '🅱️4');
                widget_x.options.forceInput = true;
                widget_xy.options.forceInput = true;
                widget_xyz.options.forceInput = true;
                widget_xyzw.options.forceInput = true;
                widget_y.options.forceInput = true;
                widget_yy.options.forceInput = true;
                widget_yyz.options.forceInput = true;
                widget_yyzw.options.forceInput = true;
                //
                const visible_x = widget_a.link === null;
                const visible_y = widget_b.link === null;
                //
                widget_hide(this, widget_x, "-jovi");
                widget_hide(this, widget_xy, "-jovi");
                widget_hide(this, widget_xyz, "-jovi");
                widget_hide(this, widget_xyzw, "-jovi");
                widget_hide(this, widget_y, "-jovi");
                widget_hide(this, widget_yy, "-jovi");
                widget_hide(this, widget_yyz, "-jovi");
                widget_hide(this, widget_yyzw, "-jovi");
                //
                if (combo.value == "BOOLEAN") {
                    show_boolean(widget_x, in_x, visible_x);
                    show_boolean(widget_y, in_y, visible_y);
                } else if (combo.value == "FLOAT") {
                    process_value(in_x, widget_x, 1, visible_x, "number", "FLOAT");
                    process_value(in_y, widget_y, 1, visible_y, "number", "FLOAT");
                } else if (combo.value == "INT") {
                    process_value(in_x, widget_x, 0, visible_x, "number", "INT");
                    process_value(in_y, widget_y, 0, visible_y, "number", "INT");
                } else if (combo.value == "VEC2INT") {
                    show_vector(widget_xy);
                    show_vector(widget_yy);
                } else if (combo.value == "VEC2") {
                    show_vector(widget_xy, 3);
                    show_vector(widget_yy, 3);
                } else if (combo.value == "VEC3INT") {
                    show_vector(widget_xyz);
                    show_vector(widget_yyz);
                } else if (combo.value == "VEC3") {
                    show_vector(widget_xyz, 3);
                    show_vector(widget_yyz, 3);
                } else if (combo.value == "VEC4INT") {
                    show_vector(widget_xyzw);
                    show_vector(widget_yyzw);
                } else if (combo.value == "VEC4") {
                    show_vector(widget_xyzw, 3);
                    show_vector(widget_yyzw, 3);
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
