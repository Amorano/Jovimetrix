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

function widgetShowVector(widget, values={}, type) {
    if (["FLOAT"].includes(type)) {
        type = "VEC1";
    } else if (["INT"].includes(type)) {
        type = "VEC1INT";
    } else if (type == "BOOLEAN") {
        type = "toggle";
    }

    if (type !== undefined) {
        widget.type = type;
    }

    if (widget.value === undefined) {
        widget.value = widget.options?.default || {};
    }

    // convert widget.value to pure dict/object
    if (Array.isArray(widget.value)) {
        let new_val = {};
        for (let i = 0; i < widget.value.length; i++) {
            new_val[i] = widget.value[i];
        }
        widget.value = new_val;
    }

    widget.options.step = 1;
    widget.options.round = 1;
    widget.options.precision = 0;
    if (widget.type != 'toggle') {
        let size = 1;
        const match = /\d/.exec(widget.type);
        if (match) {
            size = match[0];
        }
        if (!widget.type.endsWith('INT') && widget.type != 'BOOLEAN') {
            widget.options.step = 0.01;
            widget.options.round = 0.001;
            widget.options.precision = 3;
        }

        widget.value = {};
        for (let i = 0; i < size; i++) {
            widget.value[i] = (widget.options.precision == 0) ? Number(values[i]) : parseFloat(values[i]).toFixed(widget.options.precision);
        }
    } else {
        widget.value = values[0] ? true : false;
    }
}

/*
* matchFloatSize forces the target to be float[n] based on its type size
*/
export function widgetHookControl(node, control_key, child_key, matchFloatSize=false) {

    const target = node.widgets.find(w => w.name == child_key);
    const combo = node.widgets.find(w => w.name == control_key);
    if (!target || !combo) {
        throw new Error("Required widgets not found");
    }

    const initializeTrack = (widget) => {
        const track = {};
        for (let i = 0; i < 4; i++) {
            track[i] = widget.options?.default[i];
        }
        Object.assign(track, widget.value);
        return track;
    };

    const { widgets } = node;

    const data = {
        track_xyzw: initializeTrack(target),
        target,
        combo
    };

    const oldCallback = combo.callback;
    combo.callback = () => {
        const me = oldCallback?.apply(this, arguments);
        if (["VEC2", "VEC3", "VEC4", "BOOLEAN", "INT", "FLOAT"].includes(combo.value)) {
            let type = combo.value;
            if (matchFloatSize) {
                if (["BOOLEAN", "INT"].includes(combo.value)) {
                    type = "FLOAT";
                }
            }
            widgetShowVector(target, data.track_xyzw, type);
        }
        nodeFitHeight(node);
        return me;
    }

    target.callback = () => {
        if (target.type == "toggle") {
            data.track_xyzw[0] = target.value ? 1 : 0;
        } else {
            Object.keys(target.value).forEach((key) => {
                data.track_xyzw[key] = target.value[key];
            });
        }
    };

    return data;
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
export function nodeAddDynamic(nodeType, prefix, dynamic_type='*', index_start=0, match_output=false, refresh=true) {
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

        if (self?.outputs && match_output) {
            while (self.outputs.length > index_start) {
                self.removeOutput(self.outputs.length-1);
            }
        }

        if (!self.inputs) {
            return;
        }

        let idx = index_start;
        let offset = 0;
        while (idx < self.inputs.length-1) {
            const slot = self.inputs[idx];
            const parts = slot.name.split('_');
            if (parts.length == 2 && self.graph) {
                if (slot.link == null) {
                    if (match_output) {
                        self.removeOutput(idx);
                    }
                    if (idx < self.inputs.length) {
                        self.removeInput(idx);
                    }
                } else {
                    const name = parts.slice(1).join('');
                    self.inputs[idx].name = `${offset}_${name}`;
                    if (match_output) {
                        while(self.outputs.length-1 < idx) {
                            self.addOutput(prefix, dynamic_type);
                        }
                        self.outputs[idx].name = parts[1];
                        self.outputs[idx].type = self.inputs[idx].type;

                    }
                    idx += 1;
                    offset += 1;
                }
            } else {
                idx += 1;
            }

        }
    }

    index_start = Math.max(0, index_start);
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this);
        this.addInput(prefix, dynamic_type);
        return me;
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
        const me = onConnectionsChange?.apply(this, arguments);
        if (slotType == TypeSlot.Input && slot_idx >= index_start) {
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
        if (refresh) {
            clean_inputs(this);
        }
        // nodeFitHeight(this);
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
