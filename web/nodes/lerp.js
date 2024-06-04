/**
 * File: lerp
 * .js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_hide, process_value, widget_type_name, show_vector, show_boolean } from '../util/util_widget.js'

const _id = "LERP (JOV) 🔰"

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
                const widget_x = this.widgets.find(w => w.name === '🇽');
                const widget_xy = this.widgets.find(w => w.name === '🅰️2');
                const widget_xyz = this.widgets.find(w => w.name === '🅰️3');
                const widget_xyzw = this.widgets.find(w => w.name === '🅰️4');
                const widget_y = this.widgets.find(w => w.name === '🇾');
                const widget_yy = this.widgets.find(w => w.name === '🅱️2');
                const widget_yyz = this.widgets.find(w => w.name === '🅱️3');
                const widget_yyzw = this.widgets.find(w => w.name === '🅱️4');
                widget_x.options.menu = false;
                widget_xy.options.menu = false;
                widget_xyz.options.menu = false;
                widget_xyzw.options.menu = false;
                widget_y.options.menu = false;
                widget_yy.options.menu = false;
                widget_yyz.options.menu = false;
                widget_yyzw.options.menu = false;
                widget_hide(this, widget_x, "-jovi");
                widget_hide(this, widget_xy, "-jovi");
                widget_hide(this, widget_xyz, "-jovi");
                widget_hide(this, widget_xyzw, "-jovi");
                widget_hide(this, widget_y, "-jovi");
                widget_hide(this, widget_yy, "-jovi");
                widget_hide(this, widget_yyz, "-jovi");
                widget_hide(this, widget_yyzw, "-jovi");
                if (combo.value == "BOOLEAN") {
                    show_boolean(widget_x);
                    show_boolean(widget_y);
                } else if (combo.value == "FLOAT") {
                    process_value(widget_x, 3);
                    process_value(widget_y, 3);
                } else if (combo.value == "INT") {
                    process_value(widget_x);
                    process_value(widget_y);
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
