/**
 * File: text.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "TEXT GEN (JOV) 📝"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
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
            const self = this;
            autosize.callback = () => {
                widget_hide(self, cols);
                widget_hide(self, size);
                if (!autosize.value) {
                    widget_show(size);

                } else if (!letter.value) {
                    widget_show(cols);
                }
                fitHeight(self);
            }
            letter.callback = () => {
                widget_hide(self, cols);
                widget_hide(self, margin);
                widget_hide(self, spacing);
                if(!letter.value) {
                    widget_show(margin);
                    widget_show(spacing);
                    if (autosize.value) {
                        widget_show(cols);
                    }
                }
                fitHeight(self);
            }
            setTimeout(() => { autosize.callback(); }, 10);
            setTimeout(() => { letter.callback(); }, 10);
            return me;
        }
    }
})
