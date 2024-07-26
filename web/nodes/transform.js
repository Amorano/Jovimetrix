/**
 * File: transform.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'
import { hook_widget_size_mode } from '../util/util_jov.js'

const _id = "TRANSFORM (JOV) 🏝️"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            hook_widget_size_mode(this);
            const pivot = this.widgets.find(w => w.name === 'PIVOT');
            const mirror = this.widgets.find(w => w.name === '🪞');
            mirror.callback = () => {
                widgetHide(this, pivot, "-jov");
                if (mirror.value != 'NONE') {
                    widgetShow(pivot);
                }
                nodeFitHeight(this);
            }

            const tltr = this.widgets.find(w => w.name === 'TL-TR');
            const blbr = this.widgets.find(w => w.name === 'BL-BR');
            const str = this.widgets.find(w => w.name === '💪🏽');
            const proj = this.widgets.find(w => w.name === 'PROJ');
            proj.callback = () => {
                widgetHide(this, str, "-jov");
                widgetHide(this, tltr, "-jov");
                widgetHide(this, blbr, "-jov");
                if (['SPHERICAL', 'FISHEYE'].includes(proj.value)) {
                    widgetShow(str);
                } else if (['PERSPECTIVE'].includes(proj.value)) {
                    widgetShow(tltr);
                    widgetShow(blbr);
                }
                nodeFitHeight(this);
            }

            setTimeout(() => { mirror.callback(); }, 10);
            setTimeout(() => { proj.callback(); }, 10);

            const old_compute = this.getTitle;
            this.getTitle = function() {
                return `${old_compute?.apply(this)}  `;
            }
            return me;
        }
	}
})
