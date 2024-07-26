/**
 * File: shape_generator.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "SHAPE GEN (JOV) ✨"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const sides = this.widgets.find(w => w.name === 'SIDES');
            const op = this.widgets.find(w => w.name === 'SHAPE');
            op.callback = () => {
                widget_hide(this, sides, "-jov");
                if (op.value == 'POLYGON') {
                    widget_show(sides);
                }
                fitHeight(this);
            }
            setTimeout(() => { op.callback(); }, 10);
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot) {
            if (slotType === TypeSlot.Input && slot.name == 'SHAPE') {
                const widget_combo = this.widgets.find(w => w.name === 'SHAPE');
                setTimeout(() => { widget_combo.callback(); }, 10);
            }
            return onConnectionsChange?.apply(this, arguments);
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function () {
            const widget_combo = this.widgets.find(w => w.name === 'SHAPE');
            if (widget_combo.value == 'SHAPE') {
                setTimeout(() => { widget_combo.callback(); }, 10);
            }
            return onExecuted?.apply(this, arguments)
        }
	}
})
