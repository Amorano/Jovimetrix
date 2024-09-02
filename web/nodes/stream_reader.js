/**
 * File: stream_reader.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { nodeFitHeight } from '../util/util_node.js'
import{ widgetSizeModeHook2 } from '../util/util_jov.js'
import { widgetHide, widgetShow } from '../util/util_widget.js'

const _id = "STREAM READER (JOV) 📺"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return
        }

        widgetSizeModeHook2(nodeType);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const url = this.widgets.find(w => w.name === '🌐');
            const orient = this.widgets.find(w => w.name === '🧭');
            const zoom = this.widgets.find(w => w.name === '🔎');
            const dpi = this.widgets.find(w => w.name === 'DPI');
            const camera = this.widgets.find(w => w.name === '📹');
            const monitor =this.widgets.find(w => w.name === '🖥');
            const window = this.widgets.find(w => w.name === '🪟');
            const fps = this.widgets.find(w => w.name === '🏎️');
            const bbox = this.widgets.find(w => w.name === '🔲');
            const source = this.widgets.find(w => w.name === 'SRC');
            source.callback = () => {
                widgetHide(this, url);
                widgetHide(this, camera);
                widgetHide(this, monitor);
                widgetHide(this, window);
                widgetHide(this, dpi);
                widgetHide(this, bbox);
                widgetHide(this, fps);
                widgetHide(this, orient);
                widgetHide(this, zoom);

                switch (source.value) {
                    // "URL", "CAMERA", "MONITOR", "WINDOW", "SPOUT"
                    case "URL":
                        widgetShow(url);
                        break;

                    case "CAMERA":
                        widgetShow(camera);
                        widgetShow(fps);
                        widgetShow(orient);
                        widgetShow(zoom);
                        break;

                    case "MONITOR":
                        widgetShow(monitor);
                        widgetShow(bbox);
                        break;

                    case "WINDOW":
                        widgetShow(window);
                        widgetShow(dpi);
                        widgetShow(bbox);
                        break;

                    case "SPOUT":
                        widgetShow(url);
                        widgetShow(fps);
                        break;
                }
                nodeFitHeight(self);
            }
            setTimeout(() => { source.callback(); }, 10);
            return me;
        }
    }
})
