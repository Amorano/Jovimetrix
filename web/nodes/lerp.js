/**
 * File: lerp
 * .js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, TypeSlot } from '../util/util.js'
import { widget_type_name, show_vector } from '../util/util_widget.js'
import { hook_widget_AB } from '../util/util_jov.js'

const _id = "LERP (JOV) 🔰"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            const ab_data = hook_widget_AB(this, '❓');
            const callback = ab_data.combo.callback;
            ab_data.combo.callback = () => {
                callback();
                this.outputs[0].name = widget_type_name(ab_data.combo.value);
            }
            return me;
        }
        return nodeType;
	}
})
