/**
 * File: swap.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'
import { widgetOutputHookType } from '../util/util_jov.js'

const _id = "SWIZZLE (JOV) 😵"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const x = this.widgets.find(w => w.name == '🇽');
            const y = this.widgets.find(w => w.name == '🇾');
            const z = this.widgets.find(w => w.name == '🇿');
            const w = this.widgets.find(w => w.name == '🇼');
            const swap_x = this.widgets.find(w => w.name == 'SWAP X');
            const swap_y = this.widgets.find(w => w.name == 'SWAP Y');
            const swap_z = this.widgets.find(w => w.name == 'SWAP Z');
            const swap_w = this.widgets.find(w => w.name == 'SWAP W');

            widgetOutputHookType(this, '❓', 0)

            const widgets = [
                [x, swap_x],
                [y, swap_y],
                [z, swap_z],
                [w, swap_w]
            ];

            for (const [widget, swapWidget] of widgets) {
                swapWidget.callback = () => {
                    widgetHide(this, widget);
                    if (swapWidget.value == "CONSTANT") {
                        widgetShow(widget);
                    }
                    nodeFitHeight(this);
                };
                setTimeout(() => { swapWidget.callback(); }, 10);
            }
            return me;
        }
    }
})
