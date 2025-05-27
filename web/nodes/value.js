/**/

import { app } from "../../../scripts/app.js"
import { widgetHookControl, nodeFitHeight} from "../util.js"

const _id = "VALUE (JOV) 🧬"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            const me = await onNodeCreated?.apply(this, arguments);

            this.outputs[1].type = "*";
            this.outputs[2].type = "*";
            this.outputs[3].type = "*";
            this.outputs[4].type = "*";

            const ab_data = await widgetHookControl(this, 'type', 'aa');
            await widgetHookControl(this, 'type', 'bb');

            const oldCallback = ab_data.callback;
            ab_data.callback = () => {
                oldCallback?.apply(this, arguments);

                this.outputs[0].name = ab_data.value;
                this.outputs[0].type = ab_data.value;
                let type = ab_data.value;
                type = "FLOAT";
                if (ab_data.value == "INT") {
                    type = "INT";
                } else if (ab_data.value == "BOOLEAN") {
                    type = "BOOLEAN";
                }
                this.outputs[1].type = type;
                this.outputs[2].type = type;
                this.outputs[3].type = type;
                this.outputs[4].type = type;
                nodeFitHeight(this);
            }
            return me;
        }
    }
})
