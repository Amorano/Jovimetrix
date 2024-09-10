/**
 * File: filter_mask.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "FILTER MASK (JOV) 🤿"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const toggle = this.widgets.find(w => w.name == '🇴');
            const end = this.widgets.find(w => w.name == 'END');
            toggle.callback = () => {
                widgetHide(this, end);
                if (toggle.value == true) {
                    widgetShow(end);
                }
                nodeFitHeight(self);
            };
            setTimeout(() => { toggle.callback(); }, 10);
            return me;
        }
    }
})
