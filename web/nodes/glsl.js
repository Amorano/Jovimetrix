/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js"
import { api_cmd_jovian } from '../util/util_api.js'
import { widget_remove, widget_hide } from '../util/util_widget.js'
import { flashBackgroundColor } from '../util/util_fun.js'
// import { VectorWidget } from '../widget/widget_vector.js'

const _id = "GLSL (JOV) 🍩"
// uniform vec3 conversion; // 114)
// uniform ivec3 conversion2;// 0.299, 0.587, 0.114
// uniform bvec3 conversion3;//099
// uniform bvec2 conversion3;//true,false,099

const re_variable = /^uniform\s*(bool|int|float|[i|b]?vec[2-4]|mat[2-4])\s*([A-Za-z][A-Za-z0-9_]+)\s*;[\/\/\s]*\(?((?:\-?[0-9.\s,]+)+|(?:(?:true|false)\s*,?)+)/gm;

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const widget_time = this.widgets.find(w => w.name === '🕛');
            const widget_fragment = this.widgets.find(w => w.name === 'FRAGMENT');
            widget_fragment.dynamicPrompts = false;
            widget_fragment.inputEl.addEventListener('input', function (event) {
                init_fragment(event.target.value)
            });
            const widget_param = this.widgets.find(w => w.name === 'PARAM');
            widget_param.serializeValue = async () =>
                self.widgets.slice(8).reduce((result, widget) =>
                    ({ ...result, [widget.name]: widget.value }), {});
            widget_hide(this, widget_param);

            // parse this for vars... check existing vars and "types" and keep
            // or ignore as is the case -- I should stick to a common set of
            // names/types so mine don't disconnect/rename on a script change.
            function init_fragment(value) {

                let match;
                const oldWidgets = self.widgets.slice(8);
                let newWidgets = [];
                while ((match = re_variable.exec(value)) !== null) {

                    const [fullMatch, varType, varName, varValue] = match;
                    const preVal = varValue.replace(/[\n\t ]/g, '').split(',');
                    const value = preVal.map(v => {
                        if (v == true || v == "true") {
                            v = 1;
                        }
                        const vv = parseFloat(v);
                        return isNaN(v) || v === undefined || isNaN(vv) ? 0 : vv ? vv > 0 : 0;
                    });

                    let exist = newWidgets.find(w => w.name === varName);
                    const varTypeTarget = (varType === "bool") ? "toggle" : (["float", "int"].includes(varType) ? "number" : varType);

                    if (exist && exist.type !== varTypeTarget) {
                        console.info("remove", exist.type, varType, varTypeTarget);
                        widget_remove(self, exist);
                        exist = undefined;
                    }

                    let val = {default:value};
                    if (exist === undefined) {
                        if (varType == 'int') {
                            val.default = val.default[0];
                            exist = ComfyWidgets.INT(self, varName, ["INT", val], app).widget;
                        } else if (varType == 'float') {
                            val.default = val.default[0];
                            exist = ComfyWidgets.FLOAT(self, varName, ["FLOAT", val], app).widget;
                        } else if (varType == 'bool') {
                            val.default = val.default[0];
                            exist = ComfyWidgets.BOOLEAN(self, varName, ["BOOLEAN", val], app).widget;
                        } else if (varType == 'ivec2' || varType == 'vec2' || varType == 'ivec3' || varType == 'vec3' || varType == 'ivec4' || varType == 'vec4') {
                            const idx = varType[varType.length - 1];
                            val.default = val.default.slice(0, idx);
                            while (val.default.length < idx) {
                                val.default.push(0);
                            }
                            exist = this.addWidget(`VEC${idx}`, varName, val);
                            // exist = self.addCustomWidget(VectorWidget(app, varName, [`VEC${idx}`, val]));
                        }

                        if (['vec2', 'vec3', 'vec4', 'float'].includes(varType)) {
                            exist.options.step = 0.01;
                            exist.options.round = 0.00001;
                            exist.options.precision = 4;

                        } else if (['ivec2', 'ivec3', 'ivec4', 'int'].includes(varType)) {
                            exist.options.step = 1;
                            exist.options.precision = 0;
                        }

                        if (exist.options.hasOwnProperty("max")) {
                            delete exist.options.max;
                        }
                        if (exist.options.hasOwnProperty("min")) {
                            delete exist.options.min;
                        }
                        newWidgets.push(exist);
                        exist.serialize = true;
                    }
                }

                oldWidgets.forEach(old => {
                    let found = false;
                    newWidgets.forEach(w => {
                        found = (old.name === w.name);
                    });
                    if (!found) {
                        widget_remove(self, old);
                    }
                });

                const toRemove = oldWidgets.filter(x => !newWidgets.some(n => n.name === x.name));
                toRemove.forEach(widget => {
                    widget_remove(self, widget);
                });
                self.setSize(self.size);
                app.canvas.setDirty(true);
            }

            const widget_reset = this.widgets.find(w => w.name === 'RESET');
            const old_callback = widget_reset?.callback;
            widget_reset.callback = async (e) => {
                widget_reset.value = false;
                if (old_callback) {
                    old_callback(this, arguments);
                }
                api_cmd_jovian(self.id, "reset");
                widget_time.value = 0;
            }

            async function python_glsl_error(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                console.error(event.detail.e);
                await flashBackgroundColor(widget_fragment.inputEl, 250, 3, "#FF2222AA");
            }

            async function python_glsl_time(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                if (widget_time.type != "converted-widget") {
                    widget_time.value = event.detail.t;
                    app.canvas.setDirty(true);
                }
            }
            api.addEventListener("jovi-glsl-error", python_glsl_error);
            api.addEventListener("jovi-glsl-time", python_glsl_time);
            setTimeout(() => {
                init_fragment(widget_fragment.value);
            }, 10);
            return me;
        }
    }
})
