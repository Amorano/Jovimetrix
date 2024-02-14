/**
 * File: text.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight } from '../core/util.js'
import { widget_hide, widget_show } from '../core/util_widget.js'

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
            const self = this;

            const letter = this.widgets[3];
            const size = this.widgets[6];
            const align = this.widgets[7];
            const justify = this.widgets[8];
            const margin = this.widgets[9];
            const spacing = this.widgets[10];
            const auto = this.widgets[2];
            auto.callback = () => {
                widget_hide(this, letter);
                widget_hide(this, size);
                widget_hide(this, align);
                widget_hide(this, justify);
                widget_hide(this, margin);
                widget_hide(this, spacing);
                if (!auto.value) {
                    widget_show(letter);
                    widget_show(size);
                    if(!letter.value) {
                        widget_show(align);
                        widget_show(justify);
                        widget_show(margin);
                        widget_show(spacing);
                    }
                }
                fitHeight(self);
            }
            letter.callback = () => {
                widget_hide(this, size);
                widget_hide(this, align);
                widget_hide(this, justify);
                widget_hide(this, margin);
                widget_hide(this, spacing);
                if(!auto.value && !letter.value) {
                    widget_show(size);
                    widget_show(align);
                    widget_show(justify);
                    widget_show(margin);
                    widget_show(spacing);
                }
                if (letter.value) {
                    widget_show(size);
                }
                fitHeight(self);
            }
            setTimeout(() => { auto.callback(); }, 15);
            setTimeout(() => { letter.callback(); }, 15);
            return me;
        }
    }
}

app.registerExtension(ext)
