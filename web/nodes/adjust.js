/**
 * File: adjust.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "ADJUST (JOV) 🕸️"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const radius = this.widgets.find(w => w.name === '🅡');
            const amount = this.widgets.find(w => w.name === 'VAL');
            const lohi = this.widgets.find(w => w.name === 'LoHi');
            const lmh = this.widgets.find(w => w.name === 'LMH');
            const hsv = this.widgets.find(w => w.name === 'HSV');
            const contrast = this.widgets.find(w => w.name === '🌓');
            const gamma = this.widgets.find(w => w.name === '🔆');
            const op = this.widgets.find(w => w.name === '⚒️');
            op.callback = () => {
                widgetHide(this, radius);
                widgetHide(this, amount);
                widgetHide(this, lohi);
                widgetHide(this, lmh);
                widgetHide(this, hsv);
                widgetHide(this, contrast);
                widgetHide(this, gamma);
                if (["BLUR", "STACK_BLUR", "MEDIAN_BLUR", "OUTLINE"].includes(op.value)) {
                    widgetShow(radius);

                } else if (["PIXELATE", "QUANTIZE", "POSTERIZE"].includes(op.value)) {
                    widgetShow(amount);
                } else if (["HSV"].includes(op.value)) {
                    widgetShow(hsv);
                    widgetShow(gamma);
                    widgetShow(contrast);
                } else if (["LEVELS"].includes(op.value)) {
                    widgetShow(lmh);
                    widgetShow(gamma);
                } else if (["FIND_EDGES"].includes(op.value)) {
                    widgetShow(lohi);
                } else if (!["EQUALIZE"].includes(op.value)) {
                    widgetShow(radius);
                    widgetShow(amount);
                }
                nodeFitHeight(this);
            };
            setTimeout(() => { op.callback(); }, 10);
            return me;
        }
    }
})
