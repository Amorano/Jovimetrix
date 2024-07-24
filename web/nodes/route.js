/**
 * File: route.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { TypeSlot, TypeSlotEvent, fitHeight, node_add_dynamic_route } from '../util/util.js'

const _id = "ROUTE (JOV) 🚌"
const _prefix = '🔮'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        nodeType = node_add_dynamic_route(nodeType, _prefix);

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);
            if (slot_idx == 0) {
                if (event === TypeSlotEvent.Connect && slotType === TypeSlot.Input && link_info) {
                    let fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    );
                    if (fromNode) {
                        let parent_link = link_info;
                        while (parent_link !== undefined && fromNode !== undefined && ((fromNode?.inputs?.[0]?.link || fromNode?.isVirtualNode === true) || parent_link.type === "BUS")) {
                            // poorly assume there is only a single input (get/set nodes);
                            // but who is using my own TYPE other than me for its inherent purpose
                            let lastGoodNode = fromNode;
                            if (fromNode?.isVirtualNode === true) {
                                if (fromNode.findSetter) {
                                    fromNode = fromNode.findSetter(fromNode.graph);
                                    if (fromNode) {
                                        continue
                                    }
                                }
                            }
                            if (fromNode === undefined) {
                                this.inputs[0].name = `[${lastGoodNode.id}] 🚌`;
                                this.outputs = lastGoodNode.outputs;
                                break;
                            }
                            const link = fromNode.inputs[0].link;
                            parent_link = this.graph.links[link];
                            if (parent_link == undefined) {
                                break;
                            }
                            fromNode = this.graph._nodes.find(
                                (otherNode) => otherNode.id == parent_link.origin_id
                            );
                        }
                    }
                } else if (event === TypeSlotEvent.Disconnect && slotType === TypeSlot.Input) {
                    this.outputs[0].name = '🚌';
                    this.outputs = this.outputs.slice(0, 1);
                    this.inputs[0].name = '🚌';
                }
                fitHeight(this);
            }

            return me;
        }
	}
})
