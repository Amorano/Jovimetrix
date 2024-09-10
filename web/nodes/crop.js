/**
 * File: crop.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "CROP (JOV) ✂️"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const xy = this.widgets.find(w => w.name == '🇽🇾');
            const wh = this.widgets.find(w => w.name == '🇼🇭');
            const tltr = this.widgets.find(w => w.name == 'TL-TR');
            const blbr = this.widgets.find(w => w.name == 'BL-BR');
            const op = this.widgets.find(w => w.name == '⚒️');
            op.callback = () => {
                widgetHide(self, xy);
                widgetHide(self, wh);
                widgetHide(self, tltr);
                widgetHide(self, blbr);
                if (op.value == 'XY') {
                    widgetShow(xy);
                    widgetShow(wh);
                } else if (op.value == 'CENTER') {
                    widgetShow(wh);
                } else {
                    widgetShow(tltr);
                    widgetShow(blbr);
                }
                nodeFitHeight(self);
            }
            setTimeout(() => { op.callback(); }, 10);
            return me;
        }
	}
})
