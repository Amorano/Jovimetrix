/**
 * File: text.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "TEXT GEN (JOV) 📝"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const letter = this.widgets.find(w => w.name === 'LETTER');
            const cols = this.widgets.find(w => w.name === 'COLS');
            const size = this.widgets.find(w => w.name === 'SIZE');
            const margin = this.widgets.find(w => w.name === 'MARGIN');
            const spacing = this.widgets.find(w => w.name === 'SPACING');
            const autosize = this.widgets.find(w => w.name === 'AUTOSIZE');
            autosize.callback = () => {
                widgetHide(this, cols, "-jov");
                widgetHide(this, size, "-jov");
                if (!autosize.value) {
                    widgetShow(size);
                } else if (!letter.value) {
                    widgetShow(cols);
                }
                nodeFitHeight(this);
            }

            letter.callback = () => {
                widgetHide(this, cols, "-jov");
                widgetHide(this, margin, "-jov");
                widgetHide(this, spacing, "-jov");
                if(!letter.value) {
                    widgetShow(margin);
                    widgetShow(spacing);
                    if (autosize.value) {
                        widgetShow(cols);
                    }
                }
                nodeFitHeight(this);
            }

            setTimeout(() => { autosize.callback(); }, 10);
            setTimeout(() => { letter.callback(); }, 10);
            return me;
        }
    }
})
