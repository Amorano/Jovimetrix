/**
 * File: akashic.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { ComfyWidgets } from '../../../scripts/widgets.js';
import { fitHeight } from '../util/util.js'
import { escapeHtml } from '../util/util_dom.js'
import { JImageWidget } from '../widget/widget_jimage.js'

const _prefix = 'jovi'
const _id = "AKASHIC (JOV) 📓"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

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
                //console.info(new_val);
                this.message.value = new_val;
            }
            let index = 0;
            if (message.b64_images) {
                for (const img of message.b64_images) {
                    continue;
                    const w = this.addCustomWidget(
                        JImageWidget(app, `${_prefix}_${index}`, img)
                    )
                    w.parent = this;
                    index++;
                }
            }
            fitHeight(this);
        }
    }
})
