/**
 * File: value.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_show, widget_hide, process_any, process_value, widget_type_name, show_vector, show_boolean } from '../util/util_widget.js'

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
            const widget_str = this.widgets.find(w => w.name === '📝');
            const widget_x = this.widgets.find(w => w.name === 'X');
            const widget_xy = this.widgets.find(w => w.name === '🅰️2');
            const widget_xyz = this.widgets.find(w => w.name === '🅰️3');
            const widget_xyzw = this.widgets.find(w => w.name === '🅰️4');
            const widget_rng = this.widgets.find(w => w.name === 'RNG');
            const widget_y = this.widgets.find(w => w.name === 'Y');
            const widget_yy = this.widgets.find(w => w.name === '🅱️2');
            const widget_yyz = this.widgets.find(w => w.name === '🅱️3');
            const widget_yyzw = this.widgets.find(w => w.name === '🅱️4');
            const output_x = this.outputs.find(w => w.name === '🇽');
            const output_y = this.outputs.find(w => w.name === '🇾');
            const output_z = this.outputs.find(w => w.name === '🇿');
            const output_w = this.outputs.find(w => w.name === '🇼');

            widget_x.options.menu = false;
            widget_xy.options.menu = false;
            widget_xyz.options.menu = false;
            widget_xyzw.options.menu = false;
            widget_y.options.menu = false;
            widget_yy.options.menu = false;
            widget_yyz.options.menu = false;
            widget_yyzw.options.menu = false;
            widget_str.options.menu = false;

            widget_str.origComputeSize = widget_str.computeSize;
            const combo = this.widgets.find(w => w.name === '❓');

            /*
            widget_rng.callback = () => {
                widget_hide(this, widget_y, "-jovi");
                widget_hide(this, widget_yy, "-jovi");
                widget_hide(this, widget_yyz, "-jovi");
                widget_hide(this, widget_yyzw, "-jovi");
                if (combo.value == "FLOAT" && widget_rng.value === true) {
                    process_value(widget_y, 3);
                } else if (combo.value == "INT" && widget_rng.value === true) {
                    process_value(widget_y);
                } else if (combo.value == "COORD2D" && widget_rng.value === true) {
                    show_vector(widget_yy, 3);
                } else if (["VEC2", "VEC3", "VEC4"].includes(combo.value)) {
                    if (["VEC2"].includes(combo.value) && widget_rng.value === true) {
                        show_vector(widget_yy, 3);
                    } else if (["VEC3"].includes(combo.value) && widget_rng.value === true) {
                        show_vector(widget_yyz, 3);
                    } else if (["VEC4"].includes(combo.value) && widget_rng.value === true) {
                        show_vector(widget_yyzw, 3);
                    }
                } else if (["VEC2INT", "VEC3INT", "VEC4INT"].includes(combo.value)) {
                    if (["VEC2INT"].includes(combo.value) && widget_rng.value === true) {
                        show_vector(widget_yy);
                    } else if (["VEC3INT"].includes(combo.value) && widget_rng.value === true) {
                        show_vector(widget_yyz);
                    } else if (["VEC4INT"].includes(combo.value) && widget_rng.value === true) {
                        show_vector(widget_yyzw);
                    }
                }
                fitHeight(this);
            }
            */

            combo.callback = () => {
                widget_hide(this, widget_x, "-jovi");
                widget_hide(this, widget_xy, "-jovi");
                widget_hide(this, widget_xyz, "-jovi");
                widget_hide(this, widget_xyzw, "-jovi");
                widget_hide(this, widget_rng, "-jovi");
                widget_hide(this, widget_y, "-jovi");
                widget_hide(this, widget_yy, "-jovi");
                widget_hide(this, widget_yyz, "-jovi");
                widget_hide(this, widget_yyzw, "-jovi");
                widget_hide(this, widget_str, "-jovi");
                widget_str.inputEl.className = "jov-hidden";
                widget_str.computeSize = () => [0, -4];

                //
                output_x.type = "*";
                output_y.type = "*";
                output_z.type = "*";
                output_w.type = "*";
                //
                if (combo.value == "BOOLEAN") {
                    show_boolean(widget_x);
                    show_boolean(widget_y);
                    widget_show(widget_rng);
                    output_x.type = "BOOLEAN";
                } else if (combo.value == "LIST") {
                    process_any(widget_str, "LIST")
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                    output_x.type = "LIST";
                } else if (combo.value == "DICT") {
                    process_any(widget_str, "DICT")
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                    output_x.type = "DICT";
                } else if (combo.value == "ANY") {
                    process_any(widget_x, "*")
                } else if (combo.value == "MASK") {
                    process_any(widget_x, "MASK")
                    output_x.type = "MASK";
                } else if (combo.value == "STRING") {
                    process_any(widget_str, "STRING")
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                    output_x.type = "STRING";
                } else if (combo.value == "FLOAT") {
                    process_value(widget_x, 3);
                    widget_show(widget_rng);
                    if (widget_rng.value === true) {
                        process_value(widget_y, 3);
                    }
                    output_x.type = "FLOAT";
                } else if (combo.value == "INT") {
                    process_value(widget_x);
                    widget_show(widget_rng);
                    if (widget_rng.value === true) {
                        process_value(widget_y);
                    }
                    output_x.type = "INT";
                } else if (combo.value == "COORD2D") {
                    show_vector(widget_xy, 3);
                    widget_show(widget_rng);
                    if (widget_rng.value === true) {
                        show_vector(widget_yy, 3);
                    }
                    output_x.type = "FLOAT";
                    output_y.type = "FLOAT";
                } else if (["VEC2", "VEC3", "VEC4"].includes(combo.value)) {
                    output_x.type = "FLOAT";
                    widget_show(widget_rng);
                    if (["VEC2"].includes(combo.value)) {
                        show_vector(widget_xy, 3);
                        if (widget_rng.value === true) {
                            show_vector(widget_yy, 3);
                        }
                        output_y.type = "FLOAT";
                    } else if (["VEC3"].includes(combo.value)) {
                        show_vector(widget_xyz, 3);
                        if (widget_rng.value === true) {
                            show_vector(widget_yyz, 3);
                        }
                        output_z.type = "FLOAT";
                    } else if (["VEC4"].includes(combo.value)) {
                        show_vector(widget_xyzw, 3);
                        if (widget_rng.value === true) {
                            show_vector(widget_yyzw, 3);
                        }
                        output_z.type = "FLOAT";
                        output_w.type = "FLOAT";
                    }
                } else if (["VEC2INT", "VEC3INT", "VEC4INT"].includes(combo.value)) {
                    output_x.type = "INT";
                    widget_show(widget_rng);
                    if (["VEC2INT"].includes(combo.value)) {
                        show_vector(widget_xy);
                        show_vector(widget_yy);
                        output_y.type = "INT";
                    } else if (["VEC3INT"].includes(combo.value)) {
                        show_vector(widget_xyz);
                        show_vector(widget_yyz);
                        output_z.type = "INT";
                    } else if (["VEC4INT"].includes(combo.value)) {
                        output_z.type = "INT";
                        output_w.type = "INT";
                        show_vector(widget_xyzw);
                        show_vector(widget_yyzw);
                    }
                }
                this.outputs[0].name = widget_type_name(combo.value);
                fitHeight(this);
            }

            setTimeout(() => {
                combo.callback();
            }, 10);
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

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const me = onExecuted?.apply(this,arguments);
            console.info(2, message);
            let values = message["text"].toString().map(Number);
            this.outputs[1]["name"] = values[1] + " width"
            this.outputs[2]["name"] = values[2] + " height"
            this.outputs[3]["name"] = values[0] + " count"
            return me;
        }
	}
})
