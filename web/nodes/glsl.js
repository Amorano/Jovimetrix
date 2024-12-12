/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { widgetHide  } from '../util/util_widget.js';
import { flashBackgroundColor } from '../util/util_fun.js';
import{ widgetSizeModeHook } from '../util/util_jov.js'

const _id = "GLSL (JOV) 🍩";
const EVENT_JOVI_GLSL_ERROR = "jovi-glsl-error";
const EVENT_JOVI_GLSL_TIME = "jovi-glsl-time";
const RE_VARIABLE = /uniform\s+(\w+)\s+(\w+);(?:\s*\/\/\s*([0-9.,\s]*))?\s*(?:;\s*([0-9.-]+))?\s*(?:;\s*([0-9.-]+))?\s*(?:;\s*([0-9.-]+))?\s*(?:\|\s*(.*))?$/gm

app.registerExtension({
    name: 'jovimetrix.node.' + _id,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        widgetSizeModeHook(nodeType, true);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const widget_time = this.widgets.find(w => w.name == '🕛');
            const widget_vertex = this.widgets.find(w => w.name == 'VERTEX');
            const widget_fragment = this.widgets.find(w => w.name == 'FRAGMENT');
            widget_vertex.options.menu = false;
            widget_fragment.options.menu = false;
            let widget_param = this.inputs?.find(w => w.name == 'PARAM');
            if (widget_param === undefined) {
                widget_param = this.addInput('PARAM', 'JDICT');
            }
            widget_param.serializeValue = async () =>
                this.inputs.reduce((result, widget) =>
                    ({ ...result, [widget.name]: widget.value }), {});
            widgetHide(this, widget_param, '-jov');

            // parse this for vars... check existing vars and "types" and keep
            // or ignore as is the case -- I should stick to a common set of
            // names/types so mine don't disconnect/rename on a script change.
            // Parse the GLSL code for uniform variables and initialize corresponding widgets
            function shader_changed() {
                let widgets = [];
                const matches = [...widget_fragment.value.matchAll(RE_VARIABLE)];
                matches.forEach(match => {
                    const [full_match, varType, varName, varValue] = match;
                    let exist = self.inputs?.find(w => w.name == varName);
                    let type;
                    if (varType == 'int') {
                        type = "INT";
                    } else if (varType == 'float') {
                        type = "FLOAT";
                    } else if (varType == 'bool') {
                        type = "BOOLEAN";
                    } else if (varType.startsWith('ivec') || varType.startsWith('vec')) {
                        const idx = varType[varType.length - 1];
                        type = `VEC${idx}`;
                        if (varType.startsWith('ivec')) {
                            type += 'INT';
                        }
                    } else if (varType == "sampler2D") {
                        type = "IMAGE";
                    }

                    if (exist === undefined) {
                        if (["INT", "FLOAT", "BOOLEAN", "IMAGE"].includes(type)) {
                            exist = self.addInput(varName, type);
                        } else if (varType.startsWith('ivec') || varType.startsWith('vec')) {
                            const idx = varType[varType.length - 1];
                            let type = `VEC${idx}`;
                            if (varType.startsWith('ivec')) {
                                type += 'INT';
                            }
                            exist = self.addInput(varName, type);
                        }
                    } else {
                        exist.type = type;
                    }
                    exist.value = varValue;
                    widgets.push(varName);
                });

                while (self.inputs?.length > widgets.length) {
                    let idx = 0;
                    self.inputs.forEach(i => {
                        if (!widgets.includes(i.name)) {
                            self.removeInput(idx);
                        } else {
                            idx += 1;
                        }
                    })
                }
            }
            widget_fragment.inputEl.addEventListener('input', function () {
                shader_changed();
            });

            widget_vertex.inputEl.addEventListener('input', function () {
                shader_changed();
            });

            function python_glsl_error(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                console.error(event.detail.e);
                flashBackgroundColor(widget_fragment.inputEl, 250, 3, "#FF2222AA");
            }

            function python_glsl_time(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                if (!widget_time.hidden) {
                    widget_time.value = event.detail.t;
                    app.canvas.setDirty(true);
                }
            }

            api.addEventListener(EVENT_JOVI_GLSL_ERROR, python_glsl_error);
            api.addEventListener(EVENT_JOVI_GLSL_TIME, python_glsl_time);

            this.onDestroy = () => {
                api.removeEventListener(EVENT_JOVI_GLSL_ERROR, python_glsl_error);
                api.removeEventListener(EVENT_JOVI_GLSL_TIME, python_glsl_time);
            };

            setTimeout(() => { shader_changed(); }, 10);
            return me;
        }
    }
});
