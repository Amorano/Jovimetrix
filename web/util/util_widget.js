/**
 * File: util_widget.js
 * Project: Jovimetrix
 *
 */

const regex = /\d/;

const my_map = {
    STRING: "📝",
    BOOLEAN: "🇴",
    INT: "🔟",
    FLOAT: "🛟",
    VEC2: "🇽🇾",
    COORD2D: "🇽🇾",
    VEC2INT: "🇽🇾",
    VEC3: "🇽🇾\u200c🇿",
    VEC3INT: "🇽🇾\u200c🇿",
    VEC4: "🇽🇾\u200c🇿\u200c🇼",
    VEC4INT: "🇽🇾\u200c🇿\u200c🇼",
    LIST: "🧾",
    DICT: "📖",
    IMAGE: "🖼️",
    MASK: "😷",

}

export const CONVERTED_TYPE = "converted-widget"

// return the internal mapping type name
export function widget_type_name(type) { return my_map[type];}

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

export const widget_find = (widgets, name) => widgets.find(w => w.name === name);

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
    if (widget.hidden || widget.type?.startsWith(CONVERTED_TYPE + suffix)) {
        return;
    }
    widget.origType = widget.type
    widget.hidden = true;
    widget.origComputeSize = widget.computeSize;
    if (widget.serializeValue) {
        widget.origSerializeValue = widget.serializeValue;
    }
    widget.computeSize = () => [0, -4];
    widget.type = CONVERTED_TYPE + suffix;

    widget.serializeValue = () => {
		// Prevent serializing the widget if we have no input linked
		if (!node.inputs) {
			return undefined;
		}
		let node_input = node.inputs.find((i) => i.widget?.name === widget.name);

		if (!node_input || !node_input.link) {
			return undefined;
		}
		return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
    }

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            widget_hide(node, w, ':' + widget.name);
        }
    }
}

export function widget_show(widget) {
    if (widget?.origType) {
        widget.type = widget.origType;
        delete widget.origType;
    }
    if (widget?.origComputeSize) {
        widget.computeSize = widget.origComputeSize;
        delete widget.origComputeSize;
    } else {
        widget.computeSize = (target_width) => [target_width, 20];
    }

    if (widget?.origSerializeValue) {
        widget.serializeValue = widget.origSerializeValue;
        delete widget.origSerializeValue;
    }

    widget.hidden = false;
    if (widget?.linkedWidgets) {
      for (const w of widget.linkedWidgets) {
        widget_show(w)
      }
    }
}

export function show_boolean(widget_x) {
    widget_show(widget_x);
    widget_x.origType = widget_x.type;
    widget_x.type = "toggle";
}

export function show_vector(widget, values={}, type=undefined, precision=6) {
    if (["FLOAT", "INT"].includes(type)) {
        type = "VEC1";
    } else if (type == "BOOLEAN") {
        type = "toggle";
        type = "toggle";
    }
    if (type !== undefined) {
        widget.type = type;
    }
    if (widget.type == 'toggle') {
        widget.value = values[0] > 0 ? true : false;
    } else {
        let size = 1;
        const match = regex.exec(widget.type);
        if (match) {
            size = match[0];
        }
        if (widget.type.endsWith('INT')) {
            widget.options.step = 1;
            widget.options.round = 1;
            widget.options.precision = 0;
        } else {
            widget.options.step = 1 / (10^Math.max(1, precision-2));
            widget.options.round =  1 / (10^Math.max(1, precision-1));
            widget.options.precision = precision;
        }
        widget.value = {};
        for (let i = 0; i < size; i++) {
            widget.value[i] = widget.type.endsWith('INT') ? Math.round(values[i]) : Number(values[i]);
        }
    }
}

export function process_value(widget, precision=0) {
    //widget.origType = widget.type;
    widget_show(widget);
    widget.type = "number";
    if (widget?.options) {
        widget.options.precision = precision;
        if (precision == 0) {
            widget.options.step = 10;
            widget.options.round = 1;
        } else {
            widget.options.step = 1;
            widget.options.round =  0.1;
        }
    }
}

export function process_any(widget, subtype="FLOAT") {
    widget_show(widget);
    //input.type = subtype;
    if (subtype === "BOOLEAN") {
        widget.type = "toggle";
    } else if (subtype === "FLOAT" || subtype === "INT") {
        widget.type = "number";
        if (widget?.options) {
            if (subtype=="FLOAT") {
                widget.options.precision = 3;
                widget.options.step = 1;
                widget.options.round = 0.1;
            } else {
                widget.options.precision = 0;
                widget.options.step = 10;
                widget.options.round = 1;
            }
        }
    } else {
        widget.type = subtype;
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
