/**
 * File: shape_generator.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "SHAPE GENERATOR (JOV) ✨"

const ext = {
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const sides = this.widgets[1];
            const op = this.widgets[0];
            op.callback = () => {
                widget_hide(this, sides);
                if (op.value == 'POLYGON') {
                    widget_show(sides);
                }
                fitHeight(this);
            }
            setTimeout(() => { op.callback(); }, 15);
            return me;
        }
	}
}

app.registerExtension(ext)
