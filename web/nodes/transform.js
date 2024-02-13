/**
 * File: transform.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight } from '../core/util.js'
import { widget_hide, widget_show } from '../core/util_widget.js'
import { hook_widget_size_mode } from '../core/util_jov.js'

const _id = "TRANSFORM (JOV) 🏝️"

const ext = {
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;

            hook_widget_size_mode(this);

            const pivot = this.widgets[6];
            const mirror = this.widgets[5];
            mirror.callback = () => {
                widget_hide(self, pivot);
                if (mirror.value != 'NONE') {
                    widget_show(pivot);
                }
                fitHeight(self);
            }

            const tltr = this.widgets[8];
            const blbr = this.widgets[9];
            const str = this.widgets[10];
            const proj = this.widgets[7];
            proj.callback = () => {
                widget_hide(self, str);
                widget_hide(self, tltr);
                widget_hide(self, blbr);
                if (['SPHERICAL', 'FISHEYE'].includes(proj.value)) {
                    widget_show(str);
                } else if (['PERSPECTIVE'].includes(proj.value)) {
                    widget_show(tltr);
                    widget_show(blbr);
                }
                fitHeight(self);
            }

            setTimeout(() => { mirror.callback(); }, 15);
            setTimeout(() => { proj.callback(); }, 15);
            return me;
        }
	}
}

app.registerExtension(ext)
