/**
 * File: calc_binary.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_hide, widget_show, process_value, widget_type_name } from '../util/util_widget.js'

const _id = "CALC OP BINARY (JOV) 🌟"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        function show_boolean(widget_x, in_x, visible_x, typ="toggle") {
            if (!in_x && visible_x) {
                widget_show(widget_x);
                widget_x.origType = widget_x.type;
                widget_x.type = typ;
            }
        }

        function show_vector(widget, inputs, what, visible, precision=0) {
            if (!inputs.find(w => w.name === what) != undefined && visible)
            {
                widget_show(widget);
                widget.origType = widget.type;
                if (precision == 0) {
                    widget.options.step = 1;
                    widget.options.round = 0.1;
                    widget.options.precision = 0;
                } else {
                    widget.options.step = 0.1;
                    widget.options.round =  0.01;
                    widget.options.precision = 4;
                }
            }
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const combo = this.widgets.find(w => w.name === '❓');
            combo.callback = () => {
                const in_x = this.inputs.find(w => w.name === '🇽') != undefined;
                const in_y = this.inputs.find(w => w.name === '🇾') != undefined;
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
                    process_value(in_x, widget_x, 1, visible_x)
                    process_value(in_y, widget_y, 1, visible_y)
                } else if (combo.value == "INT") {
                    process_value(in_x, widget_x, 0, visible_x)
                    process_value(in_y, widget_y, 0, visible_y)
                } else if (combo.value == "VEC2INT") {
                    show_vector(widget_xy, this.inputs, '🇽🇾', visible_x);
                    show_vector(widget_yy, this.inputs, '🅱️2', visible_y);
                } else if (combo.value == "VEC2") {
                    show_vector(widget_xy, this.inputs, '🇽🇾', visible_x, 2);
                    show_vector(widget_yy, this.inputs, '🅱️2', visible_y, 2);
                } else if (combo.value == "VEC3INT") {
                    show_vector(widget_xyz, this.inputs, '🇽🇾\u200c🇿', visible_x);
                    show_vector(widget_yyz, this.inputs, '🅱️3', visible_y);
                } else if (combo.value == "VEC3") {
                    show_vector(widget_xyz, this.inputs, '🇽🇾\u200c🇿', visible_x, 2);
                    show_vector(widget_yyz, this.inputs, '🅱️3', visible_y, 2);
                } else if (combo.value == "VEC4INT") {
                    show_vector(widget_xyzw, this.inputs, '🇽🇾\u200c🇿\u200c🇼', visible_x);
                    show_vector(widget_yyzw, this.inputs, '🅱️4', visible_y);
                } else if (combo.value == "VEC4") {
                    show_vector(widget_xyzw, this.inputs, '🇽🇾\u200c🇿\u200c🇼', visible_x, 2);
                    show_vector(widget_yyzw, this.inputs, '🅱️4', visible_y, 2);
                }
                this.outputs[0].name = widget_type_name(combo.value);
                fitHeight(this);
            }
            setTimeout(() => { combo.callback(); }, 15);
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
            if (slotType === TypeSlot.Input) {
                const combo = this.widgets.find(w => w.name === '❓');
                setTimeout(() => { combo.callback(); }, 15);

            }
            return onConnectionsChange?.apply(this, arguments);
        }

        // MENU CONVERSIONS
        /*
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            // const me = getExtraMenuOptions?.apply(this, arguments);
            const combo = this.widgets.find(w => w.name === '❓');
            let toWidget = [];
            let toInput = [];
            for (const w of this.widgets) {
                if (w.options?.forceInput) {
                    continue;
                }
                if (w.type === CONVERTED_JOV_TYPE && w.hidden) {
                    toWidget.push({
                        content: `Convertz ${w.name} to widget`,
                        callback: () => {
                            convertToWidget(this, w)
                            setTimeout(() => { combo.callback(); }, 15);
                        },
                    });
                } else {
                    const config = getConfig.call(this, w.name) ?? [w.type, w.options || {}];
                    toInput.push({
                        content: `Convertz ${w.name} to input`,
                        callback: () => {
                            convertToInput(this, w, config);
                            setTimeout(() => { combo.callback(); }, 15);
                        },
                    });
                }
            }
            if (toInput.length) {
                options.push(...toInput, null);
            }

            if (toWidget.length) {
                options.push(...toWidget, null);
            }
			// return me;

		};
        */
       return nodeType;
	}
})
