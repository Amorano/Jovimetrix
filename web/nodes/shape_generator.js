/**
 * File: shape_generator.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "SHAPE GENERATOR (JOV) ✨"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const sides = this.widgets.find(w => w.name === '♾️');
            const op = this.widgets.find(w => w.name === '🇸🇴');
            op.callback = () => {
                // console.log(this.widgets);
                widget_hide(this, sides);
                if (op.value == 'POLYGON') {
                    widget_show(sides);
                }
                fitHeight(this);
            }
            setTimeout(() => { op.callback(); }, 10);
            return me;
        }
	}
})
