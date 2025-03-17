/**/

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

        nodeAddDynamic(nodeType, _prefix);

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = async function (message) {
            const me = onExecuted?.apply(this, arguments)
            if (this.widgets) {
                for (let i = 0; i < this.widgets.length; i++) {
                    this.widgets[i].onRemove?.();
                    this.widgets.splice(i, 0);
                }
                this.widgets.length = 0;
            }
            if (this.inputs.length>1) {
                for (let i = 0; i < this.inputs.length-1; i++) {
                    let textWidget = ComfyWidgets["STRING"](this, this.inputs[i].name, ["STRING", { multiline: true }], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.margin = "1px";
                    textWidget.inputEl.style.padding = "1px";
                    textWidget.inputEl.style.border = "1px";
                    textWidget.inputEl.style.backgroundColor = "#222";
                    textWidget.value = this.inputs[i].name + " ";
                    textWidget.value += message["text"][i]
                        .replace(/\\n/g, '\n')
                        .replace(/"/g, '');
                }
            }
            return me;
        }
    }
})
