/**
 * File: crop.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight, widget_hide, widget_show } from '../core/util.js'

const _id = "CROP (JOV) ✂️"

const ext = {
	name: 'jovimetrix.node.crop',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;

            const xy = this.widgets[1];
            const wh = this.widgets[2];
            const tltr = this.widgets[3];
            const blbr = this.widgets[4];
            // const rgb = this.widgets[5];
            const op = this.widgets[0];
            op.callback = () => {
                widget_hide(self, xy);
                widget_hide(self, wh);
                widget_hide(self, tltr);
                widget_hide(self, blbr);

                if (op.value == 'CORNER') {
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
}

app.registerExtension(ext)
