/**
 * File: swap.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

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
            const self = this;
            const x = this.widgets.find(w => w.name === '🇽');
            const y = this.widgets.find(w => w.name === '🇾');
            const z = this.widgets.find(w => w.name === '🇿');
            const w = this.widgets.find(w => w.name === '🇼');
            const swap_x = this.widgets.find(w => w.name === 'SWAP X');
            swap_x.callback = () => {
                widget_hide(this, x);
                if (swap_x.value == "CONSTANT") {
                    widget_show(x);
                }
                fitHeight(self);
            };
            const swap_y = this.widgets.find(w => w.name === 'SWAP Y');
            swap_y.callback = () => {
                widget_hide(this, y);
                if (swap_y.value == "CONSTANT") {
                    widget_show(y);
                }
                fitHeight(self);
            };
            const swap_z = this.widgets.find(w => w.name === 'SWAP Z');
            swap_z.callback = () => {
                widget_hide(this, z);
                if (swap_z.value == "CONSTANT") {
                    widget_show(z);
                }
                fitHeight(self);
            };
            const swap_w = this.widgets.find(w => w.name === 'SWAP W');
            swap_w.callback = () => {
                widget_hide(this, w);
                if (swap_w.value == "CONSTANT") {
                    widget_show(w);
                }
                fitHeight(self);
            };
            setTimeout(() => { swap_x.callback(); }, 0);
            setTimeout(() => { swap_y.callback(); }, 0);
            setTimeout(() => { swap_z.callback(); }, 0);
            setTimeout(() => { swap_w.callback(); }, 0);
            return me;
        }
    }
})
