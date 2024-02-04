/**
 * File: export.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'

const _id = "EXPORT (JOV) 📽"

const ext = {
	name: 'jovimetrix.node.export',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            let opt = this.widgets[4];
            let quality = this.widgets[5];
            let quality_m = this.widgets[6];
            let fps = this.widgets[7];
            let loop = this.widgets[8];
            let combo = this.widgets[3];
            combo.callback = () => {
                util.widget_hide(self, opt);
                util.widget_hide(self, quality);
                util.widget_hide(self, quality_m);
                util.widget_hide(self, fps);
                util.widget_hide(self, loop);
                switch (combo.value) {
                    case "gif":
                        util.widget_show(opt);
                        util.widget_show(fps);
                        util.widget_show(loop);
                        break;
                    case "gifski":
                        util.widget_show(quality);
                        util.widget_show(quality_m);
                        util.widget_show(fps);
                        util.widget_show(loop);
                        break;
                }
                this.setSize([this.size[0], this.computeSize([this.size[0], this.size[1]])[1]])
                this.onResize?.(this.size);
                this.setDirtyCanvas(true, true);
            }
            setTimeout(() => { combo.callback(); }, 15);
            return me;
        }
	}
}

app.registerExtension(ext)
