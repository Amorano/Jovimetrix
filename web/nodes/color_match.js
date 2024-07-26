/**
 * File: color_match.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "COLOR MATCH (JOV) 💞"

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
            const color_map = this.widgets.find(w => w.name === '🇸🇨');
            const num_color = this.widgets.find(w => w.name === 'VAL');
            const mode = this.widgets.find(w => w.name === 'MODE');
            const map = this.widgets.find(w => w.name === 'MAP');
            map.callback = () => {
                widgetHide(this, color_map, "-jov");
                widgetHide(this, num_color, "-jov");
                if (mode.value == "LUT") {
                    if (map.value == "USER_MAP") {
                        widgetShow(num_color);
                    } else {
                        widgetShow(color_map);
                    }
                }
                nodeFitHeight(self);
            };
            mode.callback = () => {
                widgetHide(this, map, "-jov");
                if (mode.value == "LUT") {
                    widgetShow(map);
                }
                setTimeout(() => { map.callback(); }, 10);
            };
            setTimeout(() => { mode.callback(); }, 10);
            return me;
        }
    }
})
