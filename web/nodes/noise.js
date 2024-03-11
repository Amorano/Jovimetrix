/**
 * File: noise.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "NOISE (JOV) 🏞"

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
            const idx = this.widgets.find(w => w.name === 'INDEX');
            const x = this.widgets.find(w => w.name === '🇽');
            const xy = this.widgets.find(w => w.name === '🇽🇾');
            //const xyz = this.widgets.find(w => w.name === '🇽🇾\u200c🇿');
            //const xyzw = this.widgets.find(w => w.name === '🇽🇾\u200c🇿\u200c🇼');
            const noise = this.widgets.find(w => w.name === 'NOISE');
            noise.callback = () => {
                widget_hide(self, idx);
                widget_hide(self, x);
                widget_hide(self, xy);
                //widget_hide(self, xyz);
                //widget_hide(self, xyzw);
                if (noise.value.endsWith('1D')) {
                    widget_show(x);
                } else {
                    widget_show(idx);
                    widget_show(xy);
                }
                fitHeight(self);
            }
            setTimeout(() => { noise.callback(); }, 15);
            return me;
        }
	}
})
