/**
 * File: pixel_swap.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "PIXEL SWAP (JOV) 🔃"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
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
                widget_hide(this, r, "-jov");
                if (swap_r.value == "CONSTANT") {
                    widget_show(r);
                }
                fitHeight(self);
            };
            const swap_g = this.widgets.find(w => w.name === 'SWAP G');
            swap_g.callback = () => {
                widget_hide(this, g, "-jov");
                if (swap_g.value == "CONSTANT") {
                    widget_show(g);
                }
                fitHeight(self);
            };
            const swap_b = this.widgets.find(w => w.name === 'SWAP B');
            swap_b.callback = () => {
                widget_hide(this, b, "-jov");
                if (swap_b.value == "CONSTANT") {
                    widget_show(b);
                }
                fitHeight(self);
            };
            const swap_a = this.widgets.find(w => w.name === 'SWAP A');
            swap_a.callback = () => {
                widget_hide(this, a, "-jov");
                if (swap_a.value == "CONSTANT") {
                    widget_show(a);
                }
                fitHeight(self);
            };
            setTimeout(() => { swap_r.callback(); }, 10);
            setTimeout(() => { swap_g.callback(); }, 10);
            setTimeout(() => { swap_b.callback(); }, 10);
            setTimeout(() => { swap_a.callback(); }, 10);
            return me;
        }
    }
})
