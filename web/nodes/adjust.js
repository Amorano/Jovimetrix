/**
 * File: adjust.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "ADJUST (JOV) 🕸️"

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
            const radius = this.widgets.find(w => w.name === '🅡');
            const amount = this.widgets.find(w => w.name === '#️⃣');
            const lohi = this.widgets.find(w => w.name === 'LoHi');
            const lmh = this.widgets.find(w => w.name === 'LMH');
            const hsv = this.widgets.find(w => w.name === '🇭🇸\u200c🇻');
            const contrast = this.widgets.find(w => w.name === '🌓');
            const gamma = this.widgets.find(w => w.name === '🔆');
            const op = this.widgets.find(w => w.name === '⚒️');
            op.callback = () => {
                widget_hide(this, radius);
                widget_hide(this, amount);
                widget_hide(this, lohi);
                widget_hide(this, lmh);
                widget_hide(this, hsv);
                widget_hide(this, contrast);
                widget_hide(this, gamma);
                if (["BLUR", "STACK_BLUR", "MEDIAN_BLUR", "OUTLINE"].includes(op.value)) {
                    widget_show(radius);
                } else if (["PIXELATE", "QUANTIZE", "POSTERIZE"].includes(op.value)) {
                    widget_show(amount);
                } else if (["HSV"].includes(op.value)) {
                    widget_show(hsv);
                    widget_show(gamma);
                    widget_show(contrast);
                } else if (["LEVELS"].includes(op.value)) {
                    widget_show(lmh);
                    widget_show(gamma);
                } else if (["FIND_EDGES"].includes(op.value)) {
                    widget_show(lohi);
                } else if (!["EQUALIZE"].includes(op.value)) {
                    widget_show(radius);
                    widget_show(amount);
                }
                fitHeight(self);
            };
            setTimeout(() => { op.callback(); }, 15);
            return me;
        }
    }
})
