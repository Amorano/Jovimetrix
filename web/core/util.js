/**
 * File: util.js
 * Project: Jovimetrix
 *
 */

export const TypeSlot = {
    Input: 1,
    Output: 2,
};

export const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

export const node_has_widgets = (node) => {
    if (!node.widgets || !node.widgets?.[Symbol.iterator]) {
      return false
    }
    return true
}

export const node_cleanup = (node) => {
    if (!node_has_widgets(node)) {
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

export function node_add_dynamic(nodeType, prefix, type='*') {
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
            // dynamic_connection(node, slot, event, `${prefix}_`, type)
            // (node, index, event, prefix='in_', type='*', names = []

            if (!this.inputs[slot].name.startsWith(prefix)) {
                return
            }

            // remove all non connected inputs
            if (event == TypeSlotEvent.Disconnect && this.inputs.length > 1) {
                // console.info(`Removing input ${slot} (${this.inputs[slot].name})`)
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
            if (this.inputs[this.inputs.length - 1].link != undefined) {
                const nextIndex = this.inputs.length
                const name = `${prefix}_${nextIndex + 1}`
                // console.info(`Adding input ${nextIndex + 1} (${name})`)
                this.addInput(name, type)
            }

            if (event === TypeSlotEvent.Connect && link_info) {
                const fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                )
                const old_type = fromNode.outputs[link_info.origin_slot].type
                this.inputs[slot].type = old_type
            } else if (event === TypeSlotEvent.Disconnect) {
                this.inputs[slot].type = type
                this.inputs[slot].label = `${prefix}_${slot + 1}`
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

        if (timeout) {
            timeout *= 1000;
            timeoutId = setTimeout(() => {
                modal.remove();
                reject(new Error("TIMEOUT"));
            }, timeout);
        }
    });
}
