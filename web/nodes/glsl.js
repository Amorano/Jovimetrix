/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { fitHeight, node_add_dynamic} from '../util/util.js'
import { widget_hide, widget_show, CONVERTED_TYPE } from '../util/util_widget.js'
import { api_cmd_jovian } from '../util/util_api.js'
import { flashBackgroundColor } from '../util/util_fun.js'

const _id = "GLSL (JOV) 🍩"
const _prefix_chan = 'iChannel'
const _prefix_var = 'iVar'

const re_variable = /^uniform\s*(bool|int|float|[i|b]?vec[2-4]|mat[2-4])\s*([A-Za-z][A-Za-z0-9_]+)\s*;[\/\/\s]*\(?((?:\-?[0-9.\s,]+)+|(?:(?:true|false)\s*,?)+)/gm;

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        nodeType = node_add_dynamic(nodeType, _prefix_chan, '*', 0, false, true, true);
        nodeType = node_add_dynamic(nodeType, _prefix_var, '*', 0, false, true, true);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);

            const widget_time = this.widgets.find(w => w.name === '🕛');
            const widget_fragment = this.widgets.find(w => w.name === 'FRAGMENT');
            widget_fragment.dynamicPrompts = false;
            widget_fragment.inputEl.addEventListener('input', function (event) {
                init_fragment(event.target.value)
            });

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
                if (widget_time.type != CONVERTED_TYPE) {
                    widget_time.value = event.detail.t;
                    app.canvas.setDirty(true);
                }
            }
            api.addEventListener("jovi-glsl-error", python_glsl_error);
            api.addEventListener("jovi-glsl-time", python_glsl_time);
            setTimeout(() => {
                // init_fragment(widget_fragment.value);
            }, 10);
            return me;
        }
    }
})
