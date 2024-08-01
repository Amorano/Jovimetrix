/**
 * File: pixel_swap.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "PIXEL SWAP (JOV) 🔃"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const r = this.widgets.find(w => w.name === '🟥');
            const g = this.widgets.find(w => w.name === '🟩');
            const b = this.widgets.find(w => w.name === '🟦');
            const a = this.widgets.find(w => w.name === '⬜');
            const swap_r = this.widgets.find(w => w.name === 'SWAP R');
            swap_r.callback = () => {
                widgetHide(this, r);
                if (swap_r.value == "CONSTANT") {
                    widgetShow(r);
                }
                nodeFitHeight(self);
            };
            const swap_g = this.widgets.find(w => w.name === 'SWAP G');
            swap_g.callback = () => {
                widgetHide(this, g);
                if (swap_g.value == "CONSTANT") {
                    widgetShow(g);
                }
                nodeFitHeight(self);
            };
            const swap_b = this.widgets.find(w => w.name === 'SWAP B');
            swap_b.callback = () => {
                widgetHide(this, b);
                if (swap_b.value == "CONSTANT") {
                    widgetShow(b);
                }
                nodeFitHeight(self);
            };
            const swap_a = this.widgets.find(w => w.name === 'SWAP A');
            swap_a.callback = () => {
                widgetHide(this, a);
                if (swap_a.value == "CONSTANT") {
                    widgetShow(a);
                }
                nodeFitHeight(self);
            };
            setTimeout(() => { swap_r.callback(); }, 10);
            setTimeout(() => { swap_g.callback(); }, 10);
            setTimeout(() => { swap_b.callback(); }, 10);
            setTimeout(() => { swap_a.callback(); }, 10);
            return me;
        }
    }
})
