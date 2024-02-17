/**
 * File: streamreader.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "STREAM READER (JOV) 📺"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;

            const url = this.widgets[1];
            const camera = this.widgets[2];
            const monitor = this.widgets[3];
            const window = this.widgets[4];
            const dpi = this.widgets[5];
            const bbox = this.widgets[6];

            const fps = this.widgets[7];
            const orient = this.widgets[13];
            const zoom = this.widgets[14];

            const mode = this.widgets[0];
            mode.callback = () => {
                widget_hide(this, url);
                widget_hide(this, camera);
                widget_hide(this, monitor);
                widget_hide(this, window);
                widget_hide(this, dpi);
                widget_hide(this, bbox);
                widget_hide(this, fps);
                widget_hide(this, orient);
                widget_hide(this, zoom);

                switch (mode.value) {
                    // "URL", "CAMERA", "MONITOR", "WINDOW"
                    case "URL":
                        widget_show(url);
                        break;

                    case "CAMERA":
                        widget_show(camera);
                        widget_show(fps);
                        widget_show(orient);
                        widget_show(zoom);
                        break;

                    case "MONITOR":
                        widget_show(monitor);
                        widget_show(bbox);
                        break;

                    case "WINDOW":
                        widget_show(window);
                        widget_show(dpi);
                        widget_show(bbox);
                        break;
                }
                fitHeight(self);
            }
            setTimeout(() => { mode.callback(); }, 15);
            return me;
        }
    }
})
