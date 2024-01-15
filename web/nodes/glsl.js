/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import * as util from '../core/util.js'

const _id = "GLSL (JOV) 🍩"
const var_regex = "uniform\s*(bool|int|float|vec[2-4])\s*([A-Za-z_]+)\s*=\s*(.*);"

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_time = this.widgets[0];
            const widget_fragment = this.widgets[6];
            widget_fragment.dynamicPrompts = false;

            widget_fragment.inputEl.addEventListener('input', function (event) {
                const value = event.target.value;
                // parse this for vars... check existing vars and "types" and keep
                // or ignore as is the case -- I should stick to a common set of
                // names/types so mine don't disconnect/rename on a script change.

            });
            const self = this;

            async function python_glsl_error(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                console.error(event.detail.e);
                await util.flashBackgroundColor(widget_fragment.inputEl, 250, 3, "#FF2222AA");
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
