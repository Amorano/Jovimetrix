/**
 * File: batcher.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight, nodeAddDynamic } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "ARRAY (JOV) 📚"
const _prefix = '❔'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }
        nodeType = nodeAddDynamic(nodeType, _prefix);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_idx = this.widgets.find(w => w.name === 'INDEX');
            const widget_range = this.widgets.find(w => w.name === 'RANGE');
            const widget_str = this.widgets.find(w => w.name === '📝');
            const widget_seed = this.widgets.find(w => w.name === 'seed');
            const widget_mode = this.widgets.find(w => w.name === 'MODE');
            const widget_count = this.widgets.find(w => w.name === 'COUNT');
            widget_mode.callback = async () => {
                widgetHide(this, widget_idx, "-jov");
                widgetHide(this, widget_range, "-jov");
                widgetHide(this, widget_str, "-jov");
                widgetHide(this, widget_seed, "-jov");
                widgetHide(this, widget_count, "-jov");
                if (widget_mode.value == "PICK") {
                    widgetShow(widget_idx);
                    widgetShow(widget_count);
                } else if (widget_mode.value == "SLICE") {
                    widgetShow(widget_range);
                } else if (widget_mode.value == "INDEX_LIST") {
                    widgetShow(widget_str);
                } else if (widget_mode.value == "RANDOM") {
                    widgetShow(widget_seed);
                    widgetShow(widget_count);
                } else if (widget_mode.value == "MERGE") {
                    // MERGE
                } else if (widget_mode.value == "CARTESIAN") {
                    console.warn("NOT IMPLEMENTED! YELL AT JOVIEX!")
                }
                nodeFitHeight(this);
            }
            setTimeout(() => { widget_mode.callback(); }, 10);
            return me;
        }
	}
})
