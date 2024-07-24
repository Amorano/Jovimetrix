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

export function isInsideRectangle(x, y, left, top, width, height) {
    if (left < x && left + width > x && top < y && top + height > y) {
        return true;
    }
    return false;
}

export function fitHeight(node) {
    const size = node.computeSize([node.size[0], node.size[1]]);
    node.setSize([node.size[0], size[1]]);
    node?.graph?.setDirtyCanvas(true);
}

/**
 * Manage the slots on a node to allow a dynamic number of inputs
*/
export function node_add_dynamic(nodeType, prefix, dynamic_type='*', index_start=0, match_output=false, refresh=true) {
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

        let idx = index_start;
        if (match_output) {
            while (self.outputs.length > index_start) {
                self.removeOutput(self.outputs.length-1);
            }
        }

        idx = index_start
        let slot_count = 0;
        while (idx < self.inputs.length-1) {
            const slot = self.inputs[idx];
            const parts = slot.name.split('_');
            if (parts.length == 2) {
                if (slot.link == null) {
                    if (match_output) {
                        self.removeOutput(idx);
                    }
                    if (idx < self.inputs.length) {
                        self.removeInput(idx);
                    }
                } else {
                    const name = parts.slice(1).join('');
                    self.inputs[idx].name = `${idx}_${name}`;
                    if (match_output) {
                        while(self.outputs.length-1 < idx) {
                            self.addOutput(prefix, dynamic_type);
                        }
                        self.outputs[idx].name = parts[1];
                        self.outputs[idx].type = self.inputs[idx].type;

                    }
                    slot_count += 1;
                    idx += 1;
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
        if (slotType === TypeSlot.Input && slot_idx >= index_start) {
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
            if (last.type != dynamic_type || last.name != prefix) {
                this.addInput(prefix, dynamic_type);
            }
        }

        if (refresh) {
            setTimeout(() => {
                clean_inputs(this);
            }, 5);
        }
        fitHeight(this);
        return me;
    }
    return nodeType;
}

/**
 * Manage the slots on a node to allow a dynamic number of inputs
*/
export function node_add_dynamic_route(nodeType, prefix) {

    const dynamic_type = "*";

    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this);
        this.addInput(prefix, dynamic_type);
        return me;
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
        const me = onConnectionsChange?.apply(this, arguments);
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
            if (last.type != dynamic_type || last.name != prefix) {
                this.addInput(prefix, dynamic_type);
            }
        }

        // fix the inputs?

        if (self.graph === undefined) {
            return;
        }

        let idx = 1;
        let slot_count = 0;
        while (idx < this.inputs.length-1) {
            const slot = this.inputs[idx];
            const parts = slot.name.split('_');
            if (parts.length == 2) {
                if (slot.link == null) {
                    this.removeOutput(idx);
                    if (idx < this.inputs.length) {
                        this.removeInput(idx);
                    }
                } else {
                    const name = parts.slice(1).join('');
                    this.inputs[idx].name = `${idx}_${name}`;
                    while(this.outputs.length-1 < idx) {
                        this.addOutput(prefix, dynamic_type);
                    }
                    this.outputs[idx].name = parts[1];
                    this.outputs[idx].type = this.inputs[idx].type;
                    slot_count += 1;
                    idx += 1;
                }
            } else {
                idx += 1;
            }
        }
        fitHeight(this);
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

/*
* wraps a single text line into maxWidth chunks
*/
function wrapText(text, maxWidth = 145) {
    const words = text.split(' ');
    const lines = [];
    let currentLine = '';
    for (const word of words) {
        const potentialLine = currentLine ? `${currentLine} ${word}` : word;
        if (potentialLine.length <= maxWidth) {
            currentLine = potentialLine;
        } else {
            if (currentLine) lines.push(currentLine);
            currentLine = word;
        }
    }
    if (currentLine) lines.push(currentLine);
    return lines;
}