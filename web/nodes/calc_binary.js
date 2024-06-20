/**
 * File: calc_binary.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_hide, process_value, widget_type_name, show_vector, show_boolean } from '../util/util_widget.js'

const _id = "OP BINARY (JOV) 🌟"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const widget_x4 = this.widgets.find(w => w.name === '🅰️🅰️');
            const widget_y4 = this.widgets.find(w => w.name === '🅱️🅱️');
            widget_x4.options.menu = false;
            widget_y4.options.menu = false;
            let bool_x = {0:false}
            let bool_y = {0:false}
            let track_xyzw = {0:0, 1:0, 2:0, 3:0};
            let track_yyzw = {0:0, 1:0, 2:0, 3:0};
            const combo = this.widgets.find(w => w.name === '❓');
            combo.callback = () => {
                const data_x = (combo.value === "BOOLEAN") ? bool_x : track_xyzw;
                const data_y = (combo.value === "BOOLEAN") ? bool_y : track_yyzw;
                show_vector(widget_x4, data_x, combo.value);
                show_vector(widget_y4, data_y, combo.value);
                this.outputs[0].name = widget_type_name(combo.value);
                fitHeight(this);
            }

            widget_x4.callback = () => {
                console.info('callback')
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
