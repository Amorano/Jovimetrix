/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import * as util from '../core/util.js'
import * as util_fun from '../core/util_fun.js'

const _id = "GLSL (JOV) 🍩"
// uniform vec3 conversion; // 114)
// uniform ivec3 conversion2;// 0.299, 0.587, 0.114
// uniform bvec3 conversion3;//099
// uniform bvec2 conversion3;//true,false,099

const re_variable = /^uniform\s*(bool|int|float|[i|b]?vec[2-4]|mat[2-4])\s*([A-Za-z][A-Za-z0-9_]+)\s*;\s*\/\/\s*\(?([0-9.\s,]+|(?:(?:true|false)\s*,?)+)/gm;

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
            const widget_fragment = this.widgets[6];
            widget_fragment.dynamicPrompts = false;
            widget_fragment.inputEl.addEventListener('input', function (event) {
                init_fragment(event.target.value)
            });

            // parse this for vars... check existing vars and "types" and keep
            // or ignore as is the case -- I should stick to a common set of
            // names/types so mine don't disconnect/rename on a script change.
            function init_fragment(value) {
                let new_widget = [];
                let match;
                while ((match = re_variable.exec(value)) !== null) {
                    const [fullMatch, varType, varName, varValue] = match;
                    const values = varValue.replace(/[\n\t ]/g, '').split(',').map(value => parseFloat(value));
                    const newWidget = {
                        name: varName,
                        type: varType,
                        value: values,
                    };
                    new_widget.push(newWidget);
                }

                // widget_fragment should be 6th widget
                const check = self.widgets.slice(7);
                check.forEach(w => {
                    const newWidget = new_widget.find(widget => widget.name === w.name && widget.type === w.type);
                    if (newWidget) {
                        w.value = newWidget.value;
                    } else {
                        util.widget_remove(self, w);
                    }
                });
            }

            const widget_reset = this.widgets[4];
            const old_callback = widget_reset?.callback;
            widget_reset.callback = async (e) => {
                widget_reset.value = false;
                if (old_callback) {
                    old_callback(this, arguments);
                }
                util.api_cmd_jovian(self.id, "reset");
                widget_time.value = 0;
            }

            async function python_glsl_error(event) {
                console.info(event.detail.id, self.id)
                if (event.detail.id != self.id) {
                    return;
                }
                console.error(event.detail.e);
                await util_fun.flashBackgroundColor(widget_fragment.inputEl, 250, 3, "#FF2222AA");
            }

            async function python_glsl_time(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                if (widget_time.type != "converted-widget") {
                    widget_time.value = event.detail.t;
                    app.graph.setDirtyCanvas(true);
                }
            }
            api.addEventListener("jovi-glsl-error", python_glsl_error);
            api.addEventListener("jovi-glsl-time", python_glsl_time);
            return me;
        }
    }
}

app.registerExtension(ext)
