/**
 * File: stack.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight, nodeAddDynamic} from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'
import { widgetSizeModeHook } from '../util/util_jov.js'

const _id = "STACK (JOV) ➕"
const _prefix = '👾'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        nodeType = nodeAddDynamic(nodeType, _prefix);
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;
            widgetSizeModeHook(this);
            const stride = this.widgets.find(w => w.name === '🦶🏽');
            const axis = this.widgets.find(w => w.name === 'AXIS');
            axis.callback = () => {
                widgetHide(self, stride, "-jov");
                if (axis.value == 'GRID') {
                    widgetShow(stride);
                }
                nodeFitHeight(self);
            }
            setTimeout(() => { axis.callback(); }, 10);
            return me;
        }
	}
})
