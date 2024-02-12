/**
 * File: util.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js"

const TYPE_HIDDEN = "hidden-"

export const TypeSlot = {
    Input: 1,
    Output: 2,
};

export const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

export async function api_get(url) {
    var response = await api.fetchApi(url, { cache: "no-store" })
    return await response.json()
}

export async function api_post(url, data) {
    return api.fetchApi(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    })
}

export async function api_cmd_jovian(id, cmd) {
    return api.fetchApi('/jovimetrix/message', {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            id: id,
            cmd: cmd
        }),
    })
}

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

export function widget_get_type(config) {
    // Special handling for COMBO so we restrict links based on the entries
    let type = config?.[0]
    let linkType = type
    if (type instanceof Array) {
      type = 'COMBO'
      linkType = linkType.join(',')
    }
    return { type, linkType }
}

export function widget_remove(node, widgetOrSlot) {
    let index = 0;
    if (typeof widgetOrSlot === 'number') {
        index = widgetOrSlot;
    }
    else if (widgetOrSlot) {
        index = node.widgets.indexOf(widgetOrSlot);
    }
    if (index > -1) {
        const w = node.widgets[index];
        if (w.canvas) {
            w.canvas.remove()
        }
        if (w.inputEl) {
            w.inputEl.remove()
        }
        w.onRemoved?.()
        node.widgets.splice(index, 1);
    }
}

export function widget_remove_all(node) {
    if (node.widgets) {
        for (const w of node.widgets) {
            widget_remove(node, w);
        }
        who.widgets.length = 0;
    }
}

export function widget_hide(node, widget, suffix = '') {
    if (widget.hidden) {
        return
    }

    widget.origType = widget.type
    widget.hidden = true
    widget.origComputeSize = widget.computeSize
    widget.origSerializeValue = widget.serializeValue
    widget.computeSize = () => [0, -4]
    widget.type = TYPE_HIDDEN + suffix
    widget.serializeValue = () => {
        // Prevent serializing the widget if we have no input linked
        try {
            const { link } = node.inputs.find((i) => i.widget?.name === widget.name)
            if (link == null || link == undefined) {
                return undefined
            }
        } catch(Exception) {

        }
        return widget.origSerializeValue
            ? widget.origSerializeValue()
            : widget.value
    }

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            widget_hide(node, w, ':' + widget.name)
        }
    }
}

export function widget_show(widget) {
    widget.type = widget.origType
    widget.computeSize = widget.origComputeSize
    widget.computeSize = (target_width) => [target_width, 20]
    widget.serializeValue = widget.origSerializeValue
    delete widget.origType
    delete widget.origComputeSize
    delete widget.origSerializeValue
    widget.hidden = false;

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
      for (const w of widget.linkedWidgets) {
        widget_show(w)
      }
    }
}

export function convertToWidget(node, widget) {
    widget_show(widget)
    const sz = node.size
    node.removeInput(node.inputs.findIndex((i) => i.widget?.name === widget.name))

    for (const widget of node.widgets) {
      widget.last_y -= LiteGraph.NODE_SLOT_HEIGHT
    }

    // Restore original size but grow if needed
    node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function convertToInput(node, widget, config) {
    // console.log(node, widget)
    widget_hide(node, widget)

    const { linkType } = widget_get_type(config)

    // Add input and store widget config for creating on primitive node
    const sz = node.size
    node.addInput(widget.name, linkType, {
      widget: { name: widget.name, config },
    })

    for (const widget of node.widgets) {
      widget.last_y += LiteGraph.NODE_SLOT_HEIGHT
    }

    // Restore original size but grow if needed
    node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export const setupDynamicConnections = (nodeType, prefix, inputType) => {
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
      this.addInput(`${prefix}_1`, inputType)
      return r
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (
      type,
      index,
      connected,
      link_info
    ) {
      const r = onConnectionsChange
        ? onConnectionsChange.apply(this, arguments)
        : undefined
      dynamic_connection(this, index, connected, `${prefix}_`, inputType)
    }
}

export const dynamic_connection = (node, index, event, prefix='in_', type='*', names = []
    ) => {
        if (!node.inputs[index].name.startsWith(prefix)) {
            return
        }
        // remove all non connected inputs
        if (event == TypeSlotEvent.Disconnect && node.inputs.length > 1) {
            console.info(`Removing input ${index} (${node.inputs[index].name})`)
            if (node.widgets) {
                const w = node.widgets.find((w) => w.name === node.inputs[index].name)
                if (w) {
                    w.onRemoved?.()
                    node.widgets.length = node.widgets.length - 1
                }
            }
            node.removeInput(index)

            // make inputs sequential again
            for (let i = 0; i < node.inputs.length; i++) {
                const name = i < names.length ? names[i] : `${prefix}${i + 1}`
                node.inputs[i].label = name
                node.inputs[i].name = name
            }
        }

        // add an extra input
        if (node.inputs[node.inputs.length - 1].link != undefined) {
            const nextIndex = node.inputs.length
            const name = nextIndex < names.length
                ? names[nextIndex]
                : `${prefix}${nextIndex + 1}`

            console.info(`Adding input ${nextIndex + 1} (${name})`)
            node.addInput(name, type)
        }
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

export function fitHeight(node) {
    node.onResize?.(node.size);
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true, true);
}
