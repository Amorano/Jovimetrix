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

    const track_options = {}
    Object.assign(track_options, target.options);

    const controlCallback = control.callback;
    control.callback = async () => {
        const me = await controlCallback?.apply(this, arguments);
        Object.assign(target.options, track_options);

        if (["VEC2", "VEC3", "VEC4", "FLOAT", "INT", "BOOLEAN"].includes(control.value)) {
            target_input.type = control.value;

            if (["INT", "FLOAT", "BOOLEAN"].includes(control.value)) {
                target.type = "VEC1";
            } else {
                target.type = control.value;
            }
            target.options.type = target.type;

            let size = 1;
            if (["VEC2", "VEC3", "VEC4"].includes(target.type)) {
                const match = /\d/.exec(target.type);
                size = match[0];
            }

            target.value = {};
            if (["VEC2", "VEC3", "VEC4", "FLOAT"].includes(control.value)) {
                for (let i = 0; i < size; i++) {
                    target.value[i] = parseFloat(track_xyzw[i]).toFixed(target.options.precision);
                }
            } else if (control.value == "INT") {
                target.options.step = 1;
                target.options.round = 0;
                target.options.precision = 0;
                target.options.int = true;

                target.value[0] = Number(track_xyzw[0]);
            } else if (control.value == "BOOLEAN") {
                target.options.step = 1;
                target.options.precision = 0;
                target.options.mij = 0;
                target.options.maj = 1;
                target.options.int = true;
                target.value[0] = track_xyzw[0] != 0 ? 1 : 0;
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

    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = async function () {
        const me = await onNodeCreated?.apply(this, arguments);

        if (this.inputs.length == 0 || this.inputs[this.inputs.length-1].name != prefix) {
            this.addInput(prefix, dynamic_type);
        }
        return me;
    }

    function slot_name(slot) {
        return slot.name.split('_');
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = async function (slotType, slot_idx, event, link_info, node_slot) {
        const me = onConnectionsChange?.apply(this, arguments);
        const slot_parts = slot_name(node_slot);
        if ((node_slot.type === dynamic_type || slot_parts.length > 1) && slotType === TypeSlot.Input && link_info !== null) {
            const fromNode = this.graph._nodes.find(
                (otherNode) => otherNode.id == link_info.origin_id
            )
            const parent_slot = fromNode.outputs[link_info.origin_slot];
            if (event === TypeSlotEvent.Connect) {
                node_slot.type = parent_slot.type;
                node_slot.name = `0_${parent_slot.name}`;
            } else {
                this.removeInput(slot_idx);
                node_slot.type = dynamic_type;
                node_slot.name = prefix;
                node_slot.link = null;
            }

            // clean off missing slot connects
            let idx = 0;
            let offset = 0;
            while (idx < this.inputs.length) {
                const parts = slot_name(this.inputs[idx]);
                if (parts.length > 1) {
                    const name = parts.slice(1).join('');
                    this.inputs[idx].name = `${offset}_${name}`;
                    offset += 1;
                }
                idx += 1;
            }
        }
        // check that the last slot is a dynamic entry....
        let last = this.inputs[this.inputs.length-1];
        if (last.type != dynamic_type || last.name != prefix) {
            this.addInput(prefix, dynamic_type);
        }
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
