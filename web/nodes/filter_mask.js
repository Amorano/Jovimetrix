/**
 * File: filter_mask.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "FILTER MASK (JOV) 🤿"

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
            const toggle = this.widgets.find(w => w.name === '🇴');
            const end = this.widgets.find(w => w.name === 'END');
            toggle.callback = () => {
                widget_hide(this, end, "-jov");
                if (toggle.value == true) {
                    widget_show(end);
                }
                fitHeight(self);
            };
            setTimeout(() => { toggle.callback(); }, 10);
            return me;
        }
    }
})
