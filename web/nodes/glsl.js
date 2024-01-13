/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import * as util from '../core/util.js'

const _id = "GLSL (JOV) 🍩"

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        let time_last;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_time = this.widgets[0];
            const widget_fragment = this.widgets[5];
            widget_fragment.dynamicPrompts = false;
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
