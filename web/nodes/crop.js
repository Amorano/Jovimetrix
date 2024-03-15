/**
 * File: crop.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "CROP (JOV) ✂️"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;
            const xy = this.widgets.find(w => w.name === '🇽🇾');
            const wh = this.widgets.find(w => w.name === '🇼🇭');
            const tltr = this.widgets.find(w => w.name === 'TL-TR');
            const blbr = this.widgets.find(w => w.name === 'BL-BR');
            const op = this.widgets.find(w => w.name === '⚒️');
            op.callback = () => {
                widget_hide(self, xy);
                widget_hide(self, wh);
                widget_hide(self, tltr);
                widget_hide(self, blbr);

                if (op.value == 'XY') {
                    widget_show(xy);
                    widget_show(wh);
                } else if (op.value == 'CENTER') {
                    widget_show(wh);
                } else {
                    widget_show(tltr);
                    widget_show(blbr);
                }
                fitHeight(self);
            }
            setTimeout(() => { op.callback(); }, 15);
            return me;
        }
	}
})
