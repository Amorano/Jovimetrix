/**/

import { app } from "../../../scripts/app.js"
import {
    TypeSlot, TypeSlotEvent, nodeFitHeight,
    nodeVirtualLinkRoot, nodeInputsClear, nodeOutputsClear
}  from "../util.js"

const _id = "ROUTE (JOV) 🚌";
const _prefix = '🔮';
const _dynamic_type = "*";

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            const me = await onNodeCreated?.apply(this, arguments);
            this.addInput(_prefix, _dynamic_type);
            nodeOutputsClear(this, 1);
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);
            let bus_connected = false;
            if (event == TypeSlotEvent.Connect && link_info) {
                let fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                );
                if (slotType == TypeSlot.Input) {
                    if (slot_idx == 0) {
                        fromNode = nodeVirtualLinkRoot(fromNode);
                        if (fromNode?.outputs && fromNode.outputs[0].type == node_slot.type) {
                            // bus connection
                            bus_connected = true;
                            nodeInputsClear(this, 1);
                            nodeOutputsClear(this, 1);
                        }
                    } else {
                        // normal connection
                        const parent_link = fromNode?.outputs[link_info.origin_slot];
                        if (parent_link) {
                            node_slot.type = parent_link.type;
                            node_slot.name = parent_link.name ; //`${fromNode.id}_${parent_link.name}`;
                            // make sure there is a matching output...
                            while(this.outputs.length < slot_idx + 1) {
                                this.addOutput(_prefix, _dynamic_type);
                            }
                            this.outputs[slot_idx].name = node_slot.name;
                            this.outputs[slot_idx].type = node_slot.type;
                        }
                    }
                }
            } else if (event == TypeSlotEvent.Disconnect) {
                bus_connected = false;
                if (slot_idx == 0) {
                    nodeInputsClear(this, 1);
                    nodeOutputsClear(this, 1);
                } else {
                    this.removeInput(slot_idx);
                    this.removeOutput(slot_idx);
                }
            }

            // add extra input if we are not in BUS connection mode
            if (!bus_connected) {
                const last = this.inputs[this.inputs.length-1];
                if (last.name != _prefix || last.type != _dynamic_type) {
                    this.addInput(_prefix, _dynamic_type);
                }
            }
            nodeFitHeight(this);
            return me;
        }

        return nodeType;
	}
})
