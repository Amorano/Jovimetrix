/**
 * File: route.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import {
    TypeSlot, TypeSlotEvent, nodeFitHeight,
    nodeVirtualLinkRoot, nodeVirtualLinkChild,
    nodeInputsClear, nodeOutputsClear
} from '../util/util_node.js'

const _id = "ROUTE (JOV) 🚌"
const _prefix = '🔮'
const _dynamic_type = "*";

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            this.addInput(_prefix, _dynamic_type);
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);
            let bus_connected = false;
            if (slot_idx === 0 && slotType === TypeSlot.Input && link_info) {
                let fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                );
                if (event === TypeSlotEvent.Connect) {
                    fromNode = nodeVirtualLinkRoot(fromNode);
                    if (fromNode) {
                        if (fromNode?.outputs && fromNode.outputs[0].type === node_slot.type) {
                            // bus connection
                            bus_connected = true;
                            nodeInputsClear(this, 1);
                            nodeOutputsClear(this, 1);
                        } else if (slot_idx > 0) {
                            // normal connection
                            const parent_link = fromNode.outputs[link_info.origin_slot];
                            if (parent_link) {
                                node_slot.type = parent_link.type;
                                node_slot.name = parent_link.name ; //`${fromNode.id}_${parent_link.name}`;
                                // make sure there is a matching output...
                                this.addOutput(node_slot.name, node_slot.type);
                            }
                        }
                    }
                }
            } else if (event === TypeSlotEvent.Disconnect) {
                bus_connected = false;
                if (slot_idx === 0) {
                    nodeInputsClear(this, 1);
                    nodeOutputsClear(this, 1);
                }
            }

            // add extra input if we are not in BUS connection mode
            if (!bus_connected) {
                const last = this.inputs[this.inputs.length-1];
                if (last.name != _prefix || last.type != _dynamic_type) {
                    this.addInput(_prefix, _dynamic_type);
                }
            }

            /*
            } else if (event === TypeSlotEvent.Disconnect && slotType === TypeSlot.Input) {

                this.outputs[0].name = '🚌';
                this.outputs = this.outputs.slice(0, 1);
                this.inputs[0].name = '🚌';
            }*/

            /*
            if (slotType === TypeSlot.Input && slot_idx > 0) {
                if (link_info && event === TypeSlotEvent.Connect) {
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    )
                    if (fromNode) {
                        const parent_link = fromNode.outputs[link_info.origin_slot];
                        if (parent_link) {
                            node_slot.type = parent_link.type;
                            node_slot.name = `_${parent_link.name}`;
                        }
                    }
                }

                // check that the last slot is a dynamic entry....
                let last = this.inputs[this.inputs.length-1];
                if (last.type != _dynamic_type || last.name != _prefix) {
                    this.addInput(_prefix, _dynamic_type);
                }
            }


            let idx = 1;
            while (idx < this.inputs.length-1) {
                const slot = this.inputs[idx];
                const parts = slot.name.split('_');
                if (parts.length != 2) {
                    idx += 1;
                    continue;
                }

                if (slot.link == null) {
                    this.removeOutput(idx);
                    if (idx < this.inputs.length) {
                        this.removeInput(idx);
                    }
                    continue;
                }

                const name = parts.slice(1).join('');
                this.inputs[idx].name = `${idx}_${name}`;
                while(this.outputs.length-1 < idx) {
                    this.addOutput(_prefix, _dynamic_type);
                }
                this.outputs[idx].name = parts[1];
                this.outputs[idx].type = this.inputs[idx].type;
                idx += 1;
            }
                */
            nodeFitHeight(this);
            return me;
        }
        return nodeType;

        /*
        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info) {
            const me = onConnectionsChange?.apply(this, arguments);
            if (slot_idx != 0) {
                return me;
            }

            if (event === TypeSlotEvent.Connect) {
                if (slotType === TypeSlot.Output) {
                    this.outputs
                } else if (slotType === TypeSlot.Input) {
                    if (link_info) {
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
                    }
                }
            } else if (event === TypeSlotEvent.Disconnect && slotType === TypeSlot.Input) {
                this.outputs[0].name = '🚌';
                this.outputs = this.outputs.slice(0, 1);
                this.inputs[0].name = '🚌';
            }
            nodeFitHeight(this);
            return me;
        }*/
	}
})
