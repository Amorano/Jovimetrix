/**
 * File: akashic.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { ComfyWidgets } from '../../../scripts/widgets.js';
import { nodeAddDynamic } from '../util/util_node.js'

const _prefix = '📥'
const _id = "AKASHIC (JOV) 📓"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        nodeType = nodeAddDynamic(nodeType, _prefix);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            this.message = ComfyWidgets.STRING(this, '', [
                    'STRING', {
                        multiline: true,
                        dynamicPrompts: false
                    },
                ], app).widget;
            this.message.value = "";
            //this.message.computeSize = () => [0, this.widgets.length * LiteGraph.NODE_TITLE_HEIGHT * 2];
            return me;
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = async function (message) {
            const me = onExecuted?.apply(this, arguments)
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
            }
            // nodeFitHeight(this);
            return me;
        }
    }
})
