/**
 * File: route.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { TypeSlot, TypeSlotEvent, node_add_dynamic } from '../util/util.js'

const _id = "ROUTE (JOV) 🚌"
const _prefix = '🔮'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        nodeType = node_add_dynamic(nodeType, _prefix, '*', 0, true);

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);
            if (slotType === TypeSlot.Input && slot_idx == 0 && link_info) {
                if (event === TypeSlotEvent.Connect) {
                    console.info("short bus added")
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    );
                    if (fromNode) {
                        const parent_link = fromNode.outputs[link_info.origin_slot];
                        if (parent_link.type == "BUS") {
                            node_slot.type = parent_link.type;
                            node_slot.name = parent_link.name;
                            const slot_out = this.outputs[slot_idx];
                            slot_out.type = parent_link.type;
                            slot_out.name = `[${parent_link.type}]`;
                        }
                    }
                } else {
                    console.info("short bus removed")
                }
            }
            return me;
        }
	}
})
