/**
 * File: text.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "TEXT GENERATOR (JOV) 📝"

const ext = {
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const letter = this.widgets[2];
            const cols = this.widgets[6];
            const size = this.widgets[7];
            const margin = this.widgets[10];
            const spacing = this.widgets[11];
            const autosize = this.widgets[3];
            autosize.callback = () => {
                widget_hide(this, cols);
                //widget_hide(this, spacing);
                widget_hide(this, size);
                if (!autosize.value) {
                    //widget_show(spacing);
                    widget_show(size);

                } else if (!letter.value) {
                    widget_show(cols);
                }
                fitHeight(this);
            }
            letter.callback = () => {
                widget_hide(this, cols);
                widget_hide(this, margin);
                widget_hide(this, spacing);
                if(!letter.value) {
                    widget_show(margin);
                    widget_show(spacing);
                    if (autosize.value) {
                        widget_show(cols);
                    }
                }
                fitHeight(this);
            }
            setTimeout(() => { autosize.callback(); }, 15);
            setTimeout(() => { letter.callback(); }, 15);
            return me;
        }
    }
}

app.registerExtension(ext)
