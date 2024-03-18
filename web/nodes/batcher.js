/**
 * File: batcher.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, node_add_dynamic } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "BATCHER (JOV) 📚"
const _prefix = 'BATCH'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        nodeType = node_add_dynamic(nodeType, _prefix);
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_mode = this.widgets.find(w => w.name === 'MODE');
            const widget_select = this.widgets.find(w => w.name === 'SELECT');
            const widget_list = this.widgets.find(w => w.name === 'AS LIST');
            const widget_chunk = this.widgets.find(w => w.name === 'CHUNK SIZE');
            const widget_xyz = this.widgets.find(w => w.name === '🇽🇾\u200c🇿');
            const widget_str = this.widgets.find(w => w.name === '📝');
            widget_mode.callback = async () => {
                widget_hide(this, widget_list);
                widget_hide(this, widget_chunk);
                widget_hide(this, widget_select);
                widget_hide(this, widget_xyz);
                widget_hide(this, widget_str);
                if (["BATCH", "MERGE", "SELECT"].includes(widget_mode.value)) {
                    widget_show(widget_list);
                    widget_show(widget_chunk);
                }
                if (widget_mode.value == "SELECT") {
                    widget_show(widget_select);
                    widget_show(widget_xyz);
                    widget_show(widget_str);
                }
                fitHeight(this);
            }
            widget_select.callback = async () => {
                fitHeight(this);
            }
            setTimeout(() => { widget_mode.callback(); }, 15);
            setTimeout(() => { widget_select.callback(); }, 15);
            return me;
        }
	}
})
