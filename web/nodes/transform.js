/**
 * File: transform.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'
import { hook_widget_size_mode } from '../util/util_jov.js'

const _id = "TRANSFORM (JOV) 🏝️"

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
            hook_widget_size_mode(this);
            const pivot = this.widgets.find(w => w.name === 'PIVOT');
            const mirror = this.widgets.find(w => w.name === '🪞');
            mirror.callback = () => {
                widget_hide(self, pivot);
                if (mirror.value != 'NONE') {
                    widget_show(pivot);
                }
                fitHeight(self);
            }
            const tltr = this.widgets.find(w => w.name === 'TL-TR');
            const blbr = this.widgets.find(w => w.name === 'BL-BR');
            const str = this.widgets.find(w => w.name === '💪🏽');
            const proj = this.widgets.find(w => w.name === 'PROJ');
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
})
