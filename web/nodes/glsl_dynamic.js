/**
 * File: glsl_dynamic.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow  } from '../util/util_widget.js';
import { apiJovimetrix } from '../util/util_api.js';

const _id = "GLSL DYNAMIC (JOV) 🧙🏽";
const EVENT_JOVI_GLSL_TIME = "jovi-glsl-time";

app.registerExtension({
    name: 'jovimetrix.node.' + _id,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.name.endsWith("(JOV) 🧙🏽")) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_time = this.widgets.find(w => w.name === '🕛');
            const widget_batch = this.widgets.find(w => w.name === 'BATCH');
            const widget_wait = this.widgets.find(w => w.name === '✋🏽');
            const widget_reset = this.widgets.find(w => w.name === 'RESET');
            const widget_fragment = this.widgets.find(w => w.name === 'FRAGMENT');
            widget_wait.options.menu = false;
            widget_reset.options.menu = false;
            widget_fragment.options.menu = false;

            widget_batch.callback = () => {
                widgetHide(this, widget_reset, '-jov');
                widgetHide(this, widget_wait, '-jov');
                if (widget_batch.value == 0) {
                    widgetShow(widget_reset);
                    widgetShow(widget_wait);
                }
                nodeFitHeight(this);
            }

            widget_reset.callback = () => {
                widget_reset.value = false;
                apiJovimetrix(this.id, "reset");
                widget_time.value = 0;
            };

            function python_glsl_time(event) {
                if (event.detail.id != this.id) {
                    return;
                }
                if (!widget_time.hidden) {
                    widget_time.value = event.detail.t;
                    app.canvas.setDirty(true);
                }
            }

            api.addEventListener(EVENT_JOVI_GLSL_TIME, python_glsl_time);
            this.onDestroy = () => {
                api.removeEventListener(EVENT_JOVI_GLSL_TIME, python_glsl_time);
            };

            setTimeout(() => { widget_batch.callback(); }, 10);
            return me;
        }
    }
});
