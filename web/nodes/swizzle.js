/**
 * File: swap.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'
import { hook_widget_type } from '../util/util_jov.js'

const _id = "SWIZZLE (JOV) 😵"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const x = this.widgets.find(w => w.name === '🇽');
            const y = this.widgets.find(w => w.name === '🇾');
            const z = this.widgets.find(w => w.name === '🇿');
            const w = this.widgets.find(w => w.name === '🇼');
            const swap_x = this.widgets.find(w => w.name === 'SWAP X');
            const swap_y = this.widgets.find(w => w.name === 'SWAP Y');
            const swap_z = this.widgets.find(w => w.name === 'SWAP Z');
            const swap_w = this.widgets.find(w => w.name === 'SWAP W');

            hook_widget_type(this, '❓', 0)

            const widgets = [
                [x, swap_x],
                [y, swap_y],
                [z, swap_z],
                [w, swap_w]
            ];

            for (const [widget, swapWidget] of widgets) {
                swapWidget.callback = () => {
                    widget_hide(this, widget, "-jov");
                    if (swapWidget.value === "CONSTANT") {
                        widget_show(widget);
                    }
                    fitHeight(this);
                };
                setTimeout(() => { swapWidget.callback(); }, 10);
            }
            return me;
        }
    }
})
