/**
 * File: export.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "EXPORT (JOV) 📽"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;
            let opt = this.widgets.find(w => w.name === 'OPT');
            let quality = this.widgets.find(w => w.name === 'QUALITY');
            let quality_m = this.widgets.find(w => w.name === 'MOTION');
            let fps = this.widgets.find(w => w.name === '🏎️');
            let loop = this.widgets.find(w => w.name === '🔄');
            let combo = this.widgets.find(w => w.name === 'FORMAT');
            combo.callback = () => {
                widget_hide(self, opt, "-jov");
                widget_hide(self, quality, "-jov");
                widget_hide(self, quality_m, "-jov");
                widget_hide(self, fps, "-jov");
                widget_hide(self, loop, "-jov");
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
                    default:
                        widget_show(opt);
                        break;
                }
                self.onResize?.(self.size);
                fitHeight(self);
            }
            setTimeout(() => { combo.callback(); }, 10);
            return me;
        }
	}
})
