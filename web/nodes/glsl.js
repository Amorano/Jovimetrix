/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js"

import { widget_remove, widget_hide, api_cmd_jovian } from '../core/util.js'
import { flashBackgroundColor } from '../core/util_fun.js'
import { VectorWidget } from '../widget/widget_vector.js'

const _id = "GLSL (JOV) 🍩"
// uniform vec3 conversion; // 114)
// uniform ivec3 conversion2;// 0.299, 0.587, 0.114
// uniform bvec3 conversion3;//099
// uniform bvec2 conversion3;//true,false,099

const re_variable = /^uniform\s*(bool|int|float|[i|b]?vec[2-4]|mat[2-4])\s*([A-Za-z][A-Za-z0-9_]+)\s*;[\/\/\s]*\(?((?:\-?[0-9.\s,]+)+|(?:(?:true|false)\s*,?)+)/gm;

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const widget_time = this.widgets[0];

            const widget_param = this.widgets[7];
            widget_param.serializeValue = async () => {
                return self.widgets.slice(8).reduce((acc, x) => ({ ...acc, [x.name]: { ...x.value.map((v, i) => ({ [i]: v })).reduce((obj, entry) => ({ ...obj, ...entry }), {}) } }), {});
            }
            widget_hide(this, widget_param);

            const widget_fragment = this.widgets[6];
            widget_fragment.dynamicPrompts = false;
            widget_fragment.inputEl.addEventListener('input', function (event) {
                init_fragment(event.target.value)
            });

            // parse this for vars... check existing vars and "types" and keep
            // or ignore as is the case -- I should stick to a common set of
            // names/types so mine don't disconnect/rename on a script change.
            function init_fragment(value) {

                let match;
                while ((match = re_variable.exec(value)) !== null) {
                    const [fullMatch, varType, varName, varValue] = match;
                    const preVal = varValue.replace(/[\n\t ]/g, '').split(',');
                    const value = preVal.map(v => {
                        const vv = parseFloat(v);
                        return isNaN(v) || v === undefined || isNaN(vv) ? 0 : vv;
                    });

                    const check = self.widgets.slice(8);
                    let exist = check.find(widget => widget.name === varName);

                    let val = {default:value, step:1};
                    if (exist === undefined || (exist && exist.type !== varType)) {
                        if (exist) {
                            widget_remove(self, exist);
                        }
                        switch (varType) {
                            case 'int':
                                val.default = val.default[0];
                                exist = ComfyWidgets.INT(self, varName, ["INT", val], app);
                                break;

                            case 'float':
                                val.step = 0.01;
                                val.precision = 6;
                                val.default = val.default[0];
                                exist = ComfyWidgets.FLOAT(self, varName, ["FLOAT", val], app);
                                break;

                            case 'bool':
                                val.default = val.default[0];
                                exist = ComfyWidgets.BOOLEAN(self, varName, ["BOOLEAN", val], app);
                                break;

                            case 'vec2':
                                val.step = 0.01;
                                val.precision = 6;
                                val.default = val.default.slice(0, 2);
                                if (val.default.length < 2) {
                                    val.default.push(val.default[0]);
                                }
                                exist = self.addCustomWidget(VectorWidget(app, varName, ["VEC2", val]));
                                break;

                            case 'ivec2':
                                val.default = val.default.slice(0, 2);
                                if (val.default.length < 2) {
                                    val.default.push(val.default[0]);
                                }
                                exist = self.addCustomWidget(VectorWidget(app, varName, ["VEC2", val]));
                                break;

                            case 'vec3':
                                val.step = 0.01;
                                val.precision = 6;
                                val.default = val.default.slice(0, 3);
                                while (val.default.length < 3) {
                                    val.default.push(val.default[val.default.length-1]);
                                }
                                exist = self.addCustomWidget(VectorWidget(app, varName, ["VEC3", val]));
                                break;

                            case 'ivec3':
                                val.default = val.default.slice(0, 3);
                                while (val.default.length < 3) {
                                    val.default.push(val.default[val.default.length-1]);
                                }
                                exist = self.addCustomWidget(VectorWidget(app, varName, ["VEC3", val]));
                                break;

                            case 'vec4':
                                val.step = 0.01;
                                val.precision = 6;
                                while (val.default.length < 4) {
                                    val.default.push(val.default[val.default.length-1]);
                                }
                                exist = self.addCustomWidget(VectorWidget(app, varName, ["VEC4", val]));
                                break;

                            case 'ivec4':
                                while (val.default.length < 4) {
                                    val.default.push(val.default[val.default.length-1]);
                                }
                                exist = self.addCustomWidget(VectorWidget(app, varName, ["VEC4", val]));
                                break;
                        }
                        check.push(exist)
                    };
                    exist.value = val.default;
                    exist.serializeValue = async() => {};
                };
                self.computeSize();
                app.canvas.setDirty(true);
            }

            const widget_reset = this.widgets[4];
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
                console.info(event.detail.id, self.id)
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
}

app.registerExtension(ext)
