/**/

import { app } from "../../scripts/app.js"
import { api } from "../../scripts/api.js"

export const TypeSlot = {
    Input: 1,
    Output: 2,
};

export const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

export async function apiJovimetrix(id, cmd, data=null, route="message", ) {
    try {
        const response = await api.fetchApi(`/cozy_comfyui/${route}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                id: id,
                cmd: cmd,
                data: data
            }),
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status} - ${response.statusText}`);
        }
        return response;

    } catch (error) {
        console.error("API call to Jovimetrix failed:", error);
        throw error;
    }
}

/*
* matchFloatSize forces the target to be float[n] based on its type size
*/
export async function widgetHookControl(node, control_key, child_key) {
    const control = node.widgets.find(w => w.name == control_key);
    const target = node.widgets.find(w => w.name == child_key);
    const target_input = node.inputs.find(w => w.name == child_key);

    if (!control || !target || !target_input) {
        throw new Error("Required widgets not found");
    }

    const track_xyzw = {
        0: target.options?.default?.[0] || 0,
        1: target.options?.default?.[1] || 0,
        2: target.options?.default?.[2] || 0,
        3: target.options?.default?.[3] || 0,
    };

    const controlCallback = control.callback;
    control.callback = async () => {
        const me = await controlCallback?.apply(this, arguments);
        if (["VEC2", "VEC3", "VEC4", "FLOAT", "INT", "BOOLEAN"].includes(control.value)) {
            target_input.type = control.value;
            //target.options.step = 0.01;
            //target.options.round = 0.001;
            //target.options.precision = 3;

            if (["INT", "FLOAT", "BOOLEAN"].includes(control.value)) {
                target.type = "VEC1";
            } else {
                target.type = control.value;
            }

            let size = 1;
            if (control.value == "INT") {
                target.options.int = true;
            } else if (["VEC2", "VEC3", "VEC4"].includes(control.value)) {
                const match = /\d/.exec(control.value);
                size = match[0];
            }

            target.value = {};
            if (["VEC2", "VEC3", "VEC4", "FLOAT"].includes(control.value)) {
                for (let i = 0; i < size; i++) {
                    target.value[i] = parseFloat(track_xyzw[i]).toFixed(target.options.precision);
                }
            } else if (control.value == "INT") {
                target.value[0] = Number(track_xyzw[0]);
            } else if (control.value == "BOOLEAN") {
                target.value[0] = track_xyzw[0] != 0 ? true : false;
            }
        }
        nodeFitHeight(node);
        return me;
    }

    const targetCallback = target.callback;
    target.callback = async () => {
        const me = await targetCallback?.apply(this, arguments);
        if (target.type == "toggle") {
            track_xyzw[0] = target.value != 0 ? 1 : 0;
        } else if (["INT", "FLOAT"].includes(target.type)) {
            track_xyzw[0] = target.value;
        } else {
            Object.keys(target.value).forEach((key) => {
                track_xyzw[key] = target.value[key];
            });
        }
        return me;
    };

    await control.callback();
    return control;
}

export function nodeFitHeight(node) {
    const size_old = node.size;
    node.computeSize();
    node.setSize([Math.max(size_old[0], node.size[0]), Math.min(size_old[1], node.size[1])]);
    node.setDirtyCanvas(!0, !1);
    app.graph.setDirtyCanvas(!0, !1);
}

/**
 * Manage the slots on a node to allow a dynamic number of inputs
*/
export async function nodeAddDynamic(nodeType, prefix, dynamic_type='*') {
    /*
    this one should just put the "prefix" as the last empty entry.
    Means we have to pay attention not to collide key names in the
    input list.

    Also need to make sure that we keep any non-dynamic ports.
    */

    // clean off missing slot connects
    function clean_inputs(self) {
        if (self.graph === undefined) {
            return;
        }

        if (!self.inputs) {
            return;
        }

        let idx = 0;
        let offset = 0;
        while (idx < self.inputs.length-1) {
            const slot = self.inputs[idx];
            const parts = slot.name.split('_');
            if (parts.length == 2 && self.graph) {
                if (slot.link == null) {
                    if (idx < self.inputs.length) {
                        self.removeInput(idx);
                    }
                } else {
                    const name = parts.slice(1).join('');
                    self.inputs[idx].name = `${offset}_${name}`;
                    idx += 1;
                    offset += 1;
                }
            } else {
                idx += 1;
            }

        }
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = async function () {
        const me = await onNodeCreated?.apply(this);
        if (this.inputs.length == 0) {
            this.addInput(prefix, dynamic_type);
        }
        return me;
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
        const me = onConnectionsChange?.apply(this, arguments);
        if (slotType == TypeSlot.Input) { //} && slot_idx >= index_start) {
            if (link_info && event == TypeSlotEvent.Connect) {
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
            if (last.type != dynamic_type || last.name != prefix) {
                this.addInput(prefix, dynamic_type);
            }
        }

        clean_inputs(this);
        nodeFitHeight(this);
        return me;
    }
}

/**
 * Trace to the root node that is not a virtual node.
 *
 * @param {Object} node - The starting node to trace from.
 * @returns {Object} - The first physical (non-virtual) node encountered, or the last node if no physical node is found.
 */
export function nodeVirtualLinkRoot(node) {
    while (node) {
        const { isVirtualNode, findSetter } = node;

        if (!isVirtualNode || !findSetter) break;
        const nextNode = findSetter(node.graph);

        if (!nextNode) break;
        node = nextNode;
    }
    return node;
}

/**
 * Trace through outputs until a physical (non-virtual) node is found.
 *
 * @param {Object} node - The starting node to trace from.
 * @returns {Object} - The first physical node encountered, or the last node if no physical node is found.
 */
function nodeVirtualLinkChild(node) {
    while (node) {
        const { isVirtualNode, findGetter } = node;

        if (!isVirtualNode || !findGetter) break;
        const nextNode = findGetter(node.graph);

        if (!nextNode) break;
        node = nextNode;
    }
    return node;
}

/**
 * Remove inputs from a node until the stop condition is met.
 *
 * @param {Array} inputs - The list of inputs associated with the node.
 * @param {number} stop - The minimum number of inputs to retain. Default is 0.
 */
export function nodeInputsClear(node, stop = 0) {
    while (node.inputs?.length > stop) {
        node.removeInput(node.inputs.length - 1);
    }
}

/**
 * Remove outputs from a node until the stop condition is met.
 *
 * @param {Array} outputs - The list of outputs associated with the node.
 * @param {number} stop - The minimum number of outputs to retain. Default is 0.
 */
export function nodeOutputsClear(node, stop = 0) {
    while (node.outputs?.length > stop) {
        node.removeOutput(node.outputs.length - 1);
    }
}
