/**
 * File: value.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { CONVERTED_TYPE, CONVERTED_JOV_TYPE, convertToWidget, widget_hide, widget_show, process_value, widget_type_name } from '../util/util_widget.js'

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
            widget_str.origComputeSize = widget_str.computeSize;
            const combo = this.widgets.find(w => w.name === '❓');
            combo.callback = () => {
                widget_str.inputEl.className = "jov-hidden";
                widget_str.computeSize = () => [0, -4];
                const in_x = this.inputs.find(w => w.name === '🇽') != undefined;
                const in_y = this.inputs.find(w => w.name === '🇾') != undefined;
                const in_z = this.inputs.find(w => w.name === '🇿') != undefined;
                const in_w = this.inputs.find(w => w.name === '🇼') != undefined;
                const widget_a = this.inputs.find(w => w.name === '🅰️');
                //
                const widget_x = this.widgets.find(w => w.name === '🇽');
                const widget_y = this.widgets.find(w => w.name === '🇾');
                const widget_z = this.widgets.find(w => w.name === '🇿');
                const widget_w = this.widgets.find(w => w.name === '🇼');
                //
                const visible = widget_a.link === null;
                widget_hide(this, widget_x, "-jovi");
                widget_hide(this, widget_y, "-jovi");
                widget_hide(this, widget_z, "-jovi");
                widget_hide(this, widget_w, "-jovi");
                widget_hide(this, widget_str, "-jovi");
                // console.info(this);
                //
                if (combo.value == "BOOLEAN") {
                    if (!in_x && visible) {
                        widget_show(widget_x);
                        widget_x.origType = widget_x.type;
                        widget_x.type = "toggle";
                    }
                } else if (combo.value == "STRING") {
                    if (!in_x && visible) {
                        widget_show(widget_str);
                        widget_str.inputEl.className = "comfy-multiline-input";
                        widget_str.computeSize = widget_str.origComputeSize;
                    }
                } else if (combo.value == "FLOAT") {
                    process_value(in_x, widget_x, 1, visible)
                } else if (combo.value == "INT") {
                    process_value(in_x, widget_x, 0, visible)
                } else if (combo.value == "VEC2") {
                    process_value(in_x, widget_x, 1, visible)
                    process_value(in_y, widget_y, 1, visible)
                } else if (combo.value == "VEC2INT") {
                    process_value(in_x, widget_x, 0, visible)
                    process_value(in_y, widget_y, 0, visible)
                } else if (combo.value == "VEC3") {
                    process_value(in_x, widget_x, 1, visible)
                    process_value(in_y, widget_y, 1, visible)
                    process_value(in_z, widget_z, 1, visible)
                } else if (combo.value == "VEC3INT") {
                    process_value(in_x, widget_x, 0, visible)
                    process_value(in_y, widget_y, 0, visible)
                    process_value(in_z, widget_z, 0, visible)
                } else if (combo.value == "VEC4") {
                    process_value(in_x, widget_x, 1, visible)
                    process_value(in_y, widget_y, 1, visible)
                    process_value(in_z, widget_z, 1, visible)
                    process_value(in_w, widget_w, 1, visible)
                } else if (combo.value == "VEC4INT") {
                    process_value(in_x, widget_x, 0, visible)
                    process_value(in_y, widget_y, 0, visible)
                    process_value(in_z, widget_z, 0, visible)
                    process_value(in_w, widget_w, 0, visible)
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

        // MENU CONVERSIONS
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const me = getExtraMenuOptions?.apply(this, arguments);
            // console.log(me)
            const combo = this.widgets.find(w => w.name === '❓');
            let toWidget = [];
            //let toInput = [];
            for (const w of this.widgets) {
                if (w.options?.forceInput) {
                    continue;
                }
                if (w.type === CONVERTED_JOV_TYPE && w.origType === CONVERTED_TYPE) {
                    toWidget.push({
                        content: `Convertz ${w.name} to widget`,
                        callback: () => {
                            convertToWidget(this, w)
                            setTimeout(() => { combo.callback(); }, 10);
                        },
                    });
                }
            }
            if (toWidget.length) {
                options.push(...toWidget, null);
            }
			return me;
		};
       return nodeType;
	}
})
