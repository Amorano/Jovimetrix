/**
 * File: stack.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, node_add_dynamic} from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'
import { hook_widget_size_mode } from '../util/util_jov.js'

const _id = "STACK (JOV) ➕"
const _prefix = '👾'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        nodeType = node_add_dynamic(nodeType, _prefix);
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;
            hook_widget_size_mode(this);
            const stride = this.widgets.find(w => w.name === '🦶🏽');
            const axis = this.widgets.find(w => w.name === 'AXIS');
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
