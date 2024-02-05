/**
 * File: text.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'
import { widget_show, widget_hide } from '../core/util.js'

const _id = "TEXT GENERATOR (JOV) 📝"

const ext = {
	name: 'jovimetrix.node.text',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;

            const align = this.widgets[6];
            const justify = this.widgets[7];
            const margin = this.widgets[8];
            const spacing = this.widgets[9];

            const single = this.widgets[5];
            single.callback = () => {
                widget_hide(this, align);
                widget_hide(this, justify);
                widget_hide(this, margin);
                widget_hide(this, spacing);
                if (!single.value) {
                    widget_show(align);
                    widget_show(justify);
                    widget_show(margin);
                    widget_show(spacing);
                }
                util.fitHeight(self);
            }
            setTimeout(() => { single.callback(); }, 15);
            return me;
        }
    }
}

app.registerExtension(ext)
