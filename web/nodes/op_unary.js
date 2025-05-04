/**/

import { app } from "../../../scripts/app.js"
import { TypeSlotEvent, TypeSlot, nodeFitHeight, widgetHookValue } from "../util.js"

const _id = "OP UNARY (JOV) 🎲"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {

        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            widgetHookValue(this, 'TYPE', 'AA');
            return me;
        }
/*
        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);
            let connected = false;
            const output = this.outputs[0];
            if (slotType == TypeSlot.Input && link_info && event == TypeSlotEvent.Connect) {
                const fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                )

                if (fromNode) {
                    const parent_link = fromNode.outputs[link_info.origin_slot];
                    if (parent_link) {
                        output.type = parent_link.type;
                        //output.name = parent_link.name;
                        //output.localized_name = parent_link.localized_name;
                        connected = true;
                    }
                }
            }
            if (connected == false) {
                output.type = "*";
                //output.name = "❔";
                //output.localized_name = "❔";
            }
            nodeFitHeight(this);
            return me;
        }
*/
       return nodeType;
	}
})

