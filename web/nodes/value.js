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
            const widget_rng = this.widgets.find(w => w.name === 'seed');
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

            const widget_combo = this.widgets.find(w => w.name === '❓');
            widget_combo.callback = () => {
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
                function set_output(what) {
                    output_x.type = what;
                    output_y.type = what;
                    output_z.type = what;
                    output_w.type = what;
                }

                //
                if (widget_combo.value == "BOOLEAN") {
                    show_boolean(widget_x);
                    widget_show(widget_rng);
                    if (widget_rng.value > 0) {
                        widget_show(widget_y);
                    }
                    set_output("BOOLEAN");
                } else if (widget_combo.value == "LIST") {
                    process_any(widget_str, "LIST")
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                    set_output("LIST");
                } else if (widget_combo.value == "DICT") {
                    process_any(widget_str, "DICT")
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                    set_output("DICT");
                } else if (widget_combo.value == "ANY") {
                    process_any(widget_x, "*")
                } else if (widget_combo.value == "MASK") {
                    process_any(widget_x, "MASK")
                    set_output("MASK");
                } else if (widget_combo.value == "STRING") {
                    process_any(widget_str, "STRING")
                    widget_str.inputEl.className = "comfy-multiline-input";
                    widget_str.computeSize = widget_str.origComputeSize;
                    set_output("STRING");
                } else if (widget_combo.value == "FLOAT") {
                    process_value(widget_x, 3);
                    widget_show(widget_rng);
                    if (widget_rng.value > 0) {
                        process_value(widget_y, 3);
                                       }
                    set_output("FLOAT");
                } else if (widget_combo.value == "INT") {
                    process_value(widget_x);
                    widget_show(widget_rng);
                    if (widget_rng.value > 0) {
                        process_value(widget_y);
                    }
                    set_output("INT");
                } else if (["VEC2", "COORD2D", "VEC3", "VEC4", "VEC2INT", "VEC3INT", "VEC4INT"].includes(widget_combo.value)) {
                    widget_show(widget_rng);
                    let precision = 0;
                    if (["FLOAT", "COORD2D", "VEC2", "VEC3", "VEC4"].includes(widget_combo.value)) {
                        precision = 3;
                        set_output("FLOAT");
                    } else {
                        set_output("INT");
                    }
                    if (["COORD2D", "VEC2", "VEC2INT"].includes(widget_combo.value)) {
                        show_vector(widget_xy, precision);
                        if (widget_rng.value > 0) {
                            show_vector(widget_yy, precision);
                        }
                    } else if (["VEC3", "VEC3INT"].includes(widget_combo.value)) {
                        show_vector(widget_xyz, precision);
                        if (widget_rng.value > 0) {
                            show_vector(widget_yyz, precision);
                        }
                    } else if (["VEC4", "VEC4INT"].includes(widget_combo.value)) {
                        show_vector(widget_xyzw, precision);
                        if (widget_rng.value > 0) {
                            show_vector(widget_yyzw, precision);
                        }
                    } else if (["INT", "FLOAT"].includes(widget_combo.value)) {
                        process_value(widget_x, precision);
                        if (widget_rng.value > 0) {
                            process_value(widget_y, precision);
                        }
                    }
                }
                this.outputs[0].name = widget_type_name(widget_combo.value);
                fitHeight(this);
            }

            widget_rng.callback = () => {
                widget_hide(this, widget_y, "-jovi");
                widget_hide(this, widget_yy, "-jovi");
                widget_hide(this, widget_yyz, "-jovi");
                widget_hide(this, widget_yyzw, "-jovi");
                if (widget_rng.value > 0) {
                    let precision = 0;
                    if (["FLOAT", "VEC2", "VEC3", "VEC4"].includes(widget_combo.value)) {
                        precision = 3;
                    }
                    if (["BOOLEAN", "INT", "FLOAT"].includes(widget_combo.value)) {
                        process_value(widget_y, precision);
                    } else if (["VEC2", "VEC2INT"].includes(widget_combo.value)) {
                        show_vector(widget_yy, precision);
                    } else if (["VEC3", "VEC3INT"].includes(widget_combo.value)) {
                        show_vector(widget_yyz, precision);
                    } else if (["VEC4", "VEC4INT"].includes(widget_combo.value)) {
                        show_vector(widget_yyzw, precision);
                    }
                }
                fitHeight(this);
            }

            setTimeout(() => {
                widget_combo.callback();
            }, 10);
            setTimeout(() => {
                widget_rng.callback();
            }, 10);
            return me;
        }

        /*
        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
            if (slotType === TypeSlot.Input) {
                const widget_combo = this.widgets.find(w => w.name === '❓');
                setTimeout(() => { widget_combo.callback(); }, 10);
            }
            return onConnectionsChange?.apply(this, arguments);
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const me = onExecuted?.apply(this,arguments);
            let values = message["text"].toString().map(Number);
            this.outputs[1]["name"] = values[1] + " width"
            this.outputs[2]["name"] = values[2] + " height"
            this.outputs[3]["name"] = values[0] + " count"
            return me;
        }*/
    }
})
