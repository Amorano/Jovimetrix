/**
 * File: export.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight, widget_hide, widget_show } from '../core/util.js'

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
            const self = this;

            let opt = this.widgets[4];
            let quality = this.widgets[5];
            let quality_m = this.widgets[6];
            let fps = this.widgets[7];
            let loop = this.widgets[8];
            let combo = this.widgets[3];
            combo.callback = () => {
                widget_hide(self, opt);
                widget_hide(self, quality);
                widget_hide(self, quality_m);
                widget_hide(self, fps);
                widget_hide(self, loop);
                switch (combo.value) {
                    case "gif":
                        widget_show(opt);
                        widget_show(fps);
                        widget_show(loop);
                        break;
                    case "gifski":
                        widget_show(quality);
                        widget_show(quality_m);
                        widget_show(fps);
                        widget_show(loop);
                        break;
                }
                self.onResize?.(self.size);
                fitHeight(self);
            }
            setTimeout(() => { combo.callback(); }, 15);
            return me;
        }
	}
}

app.registerExtension(ext)
