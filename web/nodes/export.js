/**
 * File: export.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

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
            let opt = this.widgets.find(w => w.name == 'OPT');
            let quality = this.widgets.find(w => w.name == 'QUALITY');
            let quality_m = this.widgets.find(w => w.name == 'MOTION');
            let fps = this.widgets.find(w => w.name == '🏎️');
            let loop = this.widgets.find(w => w.name == '🔄');
            let combo = this.widgets.find(w => w.name == 'FORMAT');
            combo.callback = () => {
                widgetHide(self, opt);
                widgetHide(self, quality);
                widgetHide(self, quality_m);
                widgetHide(self, fps);
                widgetHide(self, loop);
                switch (combo.value) {
                    case "gif":
                        widgetShow(opt);
                        widgetShow(fps);
                        widgetShow(loop);
                        break;
                    case "gifski":
                        widgetShow(quality);
                        widgetShow(quality_m);
                        widgetShow(fps);
                        widgetShow(loop);
                        break;
                    default:
                        widgetShow(opt);
                        break;
                }
                self.onResize?.(self.size);
                nodeFitHeight(self);
            }
            setTimeout(() => { combo.callback(); }, 10);
            return me;
        }
	}
})
