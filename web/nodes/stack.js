/**
 * File: stack.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight, widget_hide, widget_show } from '../core/util.js'
import{ hook_widget_size_mode } from '../core/util_jov.js'

const _id = "STACK (JOV) ➕"

const ext = {
	name: 'jovimetrix.node.stack',
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
}

app.registerExtension(ext)
