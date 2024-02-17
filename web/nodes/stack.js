/**
 * File: stack.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'
import { hook_widget_size_mode } from '../util/util_jov.js'

const _id = "STACK (JOV) ➕"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;

            hook_widget_size_mode(this);

            const stride = this.widgets[1];
            const axis = this.widgets[0];
            axis.callback = () => {
                widget_hide(self, stride);
                if (axis.value == 'GRID') {
                    widget_show(stride);
                }
                fitHeight(self);
            }
            setTimeout(() => { axis.callback(); }, 15);
            return me;
        }
	}
})
