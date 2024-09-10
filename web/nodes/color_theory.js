/**
 * File: color_theory.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "COLOR THEORY (JOV) 🛞"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const num_color = this.widgets.find(w => w.name == 'VAL');
            const scheme = this.widgets.find(w => w.name == 'SCHEME');
            scheme.callback = () => {
                widgetHide(this, num_color);
                if (scheme.value == "CUSTOM_TETRAD") {
                    widgetShow(num_color);
                }
            };
            setTimeout(() => { scheme.callback(); }, 10);
            return me;
        }
    }
})
