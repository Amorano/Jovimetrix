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

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
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
            api.addEventListener("jovi-glsl-error", python_glsl_error);
            return me;
        }

        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = async function () {
            console.info('hi')
            onExecuted?.apply(this, arguments);

            const widget_time = this.widgets[0];
            const widget_fps = this.widgets[1];
            const widget_batch = this.widgets[2];
            const offset = widget_fps.value / 1000 * widget_batch.value;

            widget_time.value += offset;
            app.graph.setDirtyCanvas(true);
        }
    }
}

app.registerExtension(ext)
