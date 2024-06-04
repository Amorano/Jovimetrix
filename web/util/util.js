/**
 * File: util.js
 * Project: Jovimetrix
 *
 */
/*
parse a json into a graph
const workflow = JSON.parse(json);
await this.loadGraphData(workflow);
*/

export const TypeSlot = {
    Input: 1,
    Output: 2,
};

export const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

export function getConfig(widgetName) {
	const { nodeData } = this.constructor;
	return nodeData?.input?.required[widgetName] ?? nodeData?.input?.optional?.[widgetName];
}

export const node_cleanup = (node) => {
    if (!node.widgets || !node.widgets?.[Symbol.iterator]) {
        return
    }
    for (const w of node.widgets) {
        if (w.canvas) {
            w.canvas.remove()
        }
        if (w.inputEl) {
            w.inputEl.remove()
        }
        w.onRemoved?.()
    }
}

export function node_mouse_pos(app, node) {
	return [
		app.canvas.graph_mouse[0] - node.pos[0],
		app.canvas.graph_mouse[1] - node.pos[1],
	];
}

export function fitHeight(node) {
    node.onResize?.(node.size);
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true, true);
}

export function node_add_dynamic(nodeType, prefix, type='*', count=-1) {
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this)
        this.addInput(`${prefix}_1`, type);
        return me
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
        const me = onConnectionsChange?.apply(this, arguments)
        if (slotType === TypeSlot.Input) {
            if (!this.inputs[slot].name.startsWith(prefix)) {
                return
            }

            // remove all non connected inputs
            if (event == TypeSlotEvent.Disconnect && this.inputs.length > 1) {
                if (this.widgets) {
                    const w = this.widgets.find((w) => w.name === this.inputs[slot].name)
                    if (w) {
                        w.onRemoved?.()
                        this.widgets.length = this.widgets.length - 1
                    }
                }
                this.removeInput(slot)

                // make inputs sequential again
                for (let i = 0; i < this.inputs.length; i++) {
                    const name = `${prefix}_${i + 1}`
                    this.inputs[i].label = name
                    this.inputs[i].name = name
                }
            }

            // add an extra input
            if (count-1 < 0) {
                count = 1000;
            }
            const length = this.inputs.length - 1;
            if (length < count-1 && this.inputs[length].link != undefined) {
                const nextIndex = this.inputs.length
                const name = `${prefix}_${nextIndex + 1}`
                this.addInput(name, type)
            }

            if (event === TypeSlotEvent.Connect && link_info) {
                const fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                )
                if (fromNode) {
                    const old_type = fromNode.outputs[link_info.origin_slot].type;
                    this.inputs[slot].type = old_type;
                }
            } else if (event === TypeSlotEvent.Disconnect) {
                this.inputs[slot].type = type
                this.inputs[slot].label = `${prefix}_${slot + 1}`
            }
        }
        return me;
    }
    return nodeType;
}

export function node_add_dynamic2(nodeType, prefix, dynamic_type='*', index_start=0, shape=LiteGraph.GRID_SHAPE) {
    /*
    this one should just put the "prefix" as the last empty entry.
    Means we have to pay attention not to collide key names in the
    input list.
    */
    index_start = Math.max(0, index_start);
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this);
        this.addInput(prefix, dynamic_type);
		this.addOutput(prefix, dynamic_type, { shape: shape });
        return me;
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
        const me = onConnectionsChange?.apply(this, arguments)
        if (slotType === TypeSlot.Input) {
            slot = this.inputs[slot];
            const who = slot.name;
            const wIndex = this.inputs.findIndex((w) => w.name === who);
            if (wIndex >= index_start) {
                let wo = this.outputs[wIndex];
                if (event == TypeSlotEvent.Disconnect) {
                    this.removeOutput(wo);
                    this.removeInput(slot);
                } else if (event === TypeSlotEvent.Connect && link_info) {
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    )
                    if (fromNode) {
                        const parent_link = fromNode.outputs[link_info.origin_slot];
                        slot.type = parent_link.type;
                        slot.name = parent_link.name;
                        wo.type = parent_link.type;
                        wo.name = `[${parent_link.type}]`;
                        this.addInput(prefix, dynamic_type);
                        this.addOutput(prefix, dynamic_type);
                    }
                }
            }
        }
        return me;
    }
    return nodeType;
}

export function convertArrayToObject(values, length, parseFn) {
    const result = {};
    for (let i = 0; i < length; i++) {
        result[i] = parseFn(values[i]);
    }
    return result;
}

export function showModal(innerHTML, eventCallback, timeout=null) {
    return new Promise((resolve, reject) => {
        const modal = document.createElement("div");
        modal.className = "modal";
        modal.innerHTML = innerHTML;
        document.body.appendChild(modal);

        // center
        const modalContent = modal.querySelector(".jov-modal-content");
        modalContent.style.position = "absolute";
        modalContent.style.left = "50%";
        modalContent.style.top = "50%";
        modalContent.style.transform = "translate(-50%, -50%)";

        let timeoutId;

        const handleEvent = (event) => {
            const targetId = event.target.id;
            const result = eventCallback(targetId);

            if (result != null) {
                if (timeoutId) {
                    clearTimeout(timeoutId);
                    timeoutId = null;
                }
                modal.remove();
                resolve(result);
            }
        };
        modalContent.addEventListener("click", handleEvent);
        modalContent.addEventListener("dblclick", handleEvent);

        if (timeout) {
            timeout *= 1000;
            timeoutId = setTimeout(() => {
                modal.remove();
                reject(new Error("TIMEOUT"));
            }, timeout);
        }

        //setTimeout(() => {
        //    modal.dispatchEvent(new Event('tick'));
        //}, 1000);
    });
}
