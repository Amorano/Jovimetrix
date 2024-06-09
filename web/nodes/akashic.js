/**
 * File: akashic.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { ComfyWidgets } from '../../../scripts/widgets.js';
import { JImageWidget } from '../widget/widget_jimage.js'
import { node_add_dynamic } from '../util/util.js'

const _prefix = '📥'
const _prefix_image = 'jovi'
const _id = "AKASHIC (JOV) 📓"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        nodeType = node_add_dynamic(nodeType, _prefix);
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            this.message = ComfyWidgets.STRING(this, '', [
                    'STRING', {
                        multiline: true,
                    },
                ], app).widget;
            this.message.value = "";
            return me;
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments)
            let lineCount = 0;
            if (this.widgets) {
                for (let i = 2; i < this.widgets.length; i++) {
                    if (this.widgets[i].name.startsWith("jovi_")) {
                        this.widgets[i].onRemoved?.();
                    }
                }
            }

            this.message.value = "";
            if (message.text != null) {
                let new_val = message.text.map((txt, index) => `${index}: ${txt}`).join('\n');
                this.message.value = new_val;
                for (let char of new_val) {
                    if (char === '\n') {
                        lineCount++;
                    }
                }
            }
            let index = 0;
            if (message.b64_images) {
                for (const img of message.b64_images) {
                    continue;
                    const w = this.addCustomWidget(
                        JImageWidget(app, `${_prefix_image}_${index}`, img)
                    )
                    w.parent = this;
                    index++;
                }
            }
            //this.onResize?.(this.size);
            //const y = this.computeSize([this.size[0], this.size[1]])[1];
            //this.setSize([this.size[0], Math.min(250, y+lineCount * 17)]);
            //this?.graph?.setDirtyCanvas(true, true);
        }
    }
})
