/**
 * File: color_theory.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "COLOR THEORY (JOV) 🛞"

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
            const num_color = this.widgets.find(w => w.name === '#️⃣');
            const scheme = this.widgets.find(w => w.name === 'SCHEME');
            scheme.callback = () => {
                // debugger;
                widget_hide(this, num_color);
                if (scheme.value == "CUSTOM_TETRAD") {
                    widget_show(num_color);
                }
                fitHeight(self);
            };
            setTimeout(() => { scheme.callback(); }, 15);
            return me;
        }
    }
})
