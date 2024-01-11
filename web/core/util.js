/**
 * File: util.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { api } from "/scripts/api.js"

const CONVERTED_TYPE = "converted-widget";

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

export async function local_get(url, d) {
    const v = localStorage.getItem(url)
    if (v && !isNaN(+v)) {
        return v
    }
    return d
}

export async function local_set(url, v) {
    localStorage.setItem(url, v)
}

export function setting_make(id, name, type, tip, value, callback=undefined) {
    app.ui.settings.addSetting({
        id: id,
        name: name,
        type: type,
        tooltip: tip,
        defaultValue: value,
        onChange(v) {
            if (callback !== undefined) {
                callback(v)
            }
        },
    })
}

export function showModal(innerHTML, eventCallback, timeout = null) {
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

export let NODE_LIST = await api_get("./../object_info")
export let CONFIG_CORE = await api_get("/jovimetrix/config")
export let CONFIG_USER = CONFIG_CORE.user.default
export let CONFIG_COLOR = CONFIG_USER.color
export let CONFIG_REGEX = CONFIG_COLOR.regex || []
export let CONFIG_THEME = CONFIG_COLOR.theme
export let USER = 'user.default'

// gets the CONFIG entry for name
export function node_color_get(node) {
    const find_me = node.type || node.name;
    if (find_me === undefined) {
        return
    }
    // First look to regex....
    for (const colors of CONFIG_REGEX) {
        if (colors.regex == "") {
            continue
        }
        const regex = new RegExp(colors.regex, 'i');
        const found = find_me.match(regex);
        if (found !== null && found[0].length > 0) {
            console.debug("node_color_get", colors, found, node)
            colors.jov_set_color = 1;
            colors.jov_set_bgcolor = 1;
            return colors;
        }
    }
    // now look to theme
    let color = CONFIG_THEME[find_me]
    if (color) {
        return color
    }
    color = NODE_LIST[find_me]
    // now look to category theme
    if (color && color.category) {
        const segments = color.category.split('/')
        let k = segments.join('/')
        while (k) {
            const found = CONFIG_THEME[k]
            if (found) {
                return found
            }
            const last = k.lastIndexOf('/')
            k = last !== -1 ? k.substring(0, last) : ''
        }
    }
    // if we made it here, we could have "temp" colors. reset.
    if (node?.jov_set_color == 1)
    {
        node.color = ""
    }
    if (node?.jov_set_bgcolor == 1)
    {
        node.bgcolor = ""
    }
    return null;
}

// refresh the color of a node
export function node_color_reset(node, refresh=true) {
    const data = node_color_get(node)
    if (data) {
        node.bgcolor = data.body
        node.color = data.title
        // console.info(node, data)
        if (refresh) {
            node.setDirtyCanvas(true, true)
        }
    }
}

export function node_color_list(nodes) {
    Object.entries(nodes).forEach((node) => {
        node_color_reset(node, false)
    })
    app.graph.setDirtyCanvas(true, true)
}

export function node_color_all() {
    app.graph._nodes.forEach((node) => {
        this.node_color_reset(node, false)
    })
    app.graph.setDirtyCanvas(true, true)
    // console.info("JOVI] all nodes color refreshed")
}

export function renderTemplate(template, data) {
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g')
            template = template.replace(regex, data[key])
        }
    }
    return template
}

export function convert_hex(color) {
    if (!color.HEX.includes("NAN")) {
        return '#' + color.HEX + ((color.alpha * 255) | 1 << 8).toString(16).slice(1).toUpperCase()
    }
    return "#353535FF"
}

export function hexToRgb(hex) {
  //console.info(hex)
  hex = hex.replace(/^#/, '');

  // Parse the hex value into RGB components
  const bigint = parseInt(hex, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;

  return [r, g, b];
}

export function rgbToHex(rgb) {
  // Convert RGB components to a hex color string
  const [r, g, b] = rgb;
  return `#${(1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1)}`;
}

export function fade_lerp_color(colorStart, colorEnd, lerp) {
  // Parse color strings into RGB arrays
  const startRGB = hexToRgb(colorStart);
  const endRGB = hexToRgb(colorEnd);

  // Linearly interpolate each RGB component
  const lerpedRGB = startRGB.map((component, index) => {
      return Math.round(component + (endRGB[index] - component) * lerp);
  });

  // Convert the interpolated RGB values back to a hex color string
  return rgbToHex(lerpedRGB);
}

export function color_contrast(hexColor) {
    // Remove the # symbol if it exists
    if (hexColor.startsWith("#")) {
        hexColor = hexColor.slice(1);
    }

    // Expand short hex code to full hex code
    if (hexColor.length === 3) {
        hexColor = hexColor.split('').map(char => char + char).join('');
    }

    // Convert the hex values to decimal (base 10) integers
    const r = parseInt(hexColor.slice(0, 2), 16) / 255;
    const g = parseInt(hexColor.slice(2, 4), 16) / 255;
    const b = parseInt(hexColor.slice(4, 6), 16) / 255;

    // Calculate the relative luminance
    const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // Use the contrast ratio to determine the text color
    return L > 0.210 ? "#000000" : "#999999";
}

export function toggleFoldable(elementId, symbolId) {
    const content = document.getElementById(elementId)
    const symbol = document.getElementById(symbolId)
    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'flex'
        symbol.innerHTML = '&#9661' // Down arrow
    } else {
        content.style.display = 'none'
        symbol.innerHTML = '&#9655' // Right arrow
    }
}

function convertArrayToObject(values, length, parseFn) {
    const result = {};
    for (let i = 0; i < length; i++) {
        result[i] = parseFn(values[i]);
    }
    return result;
}

export const SlotType = {
    Input: 1,
    Output: 2,
};

export const ChangeType = {
    Connect: true,
    Disconnect: false,
};

export function inner_value_change(widget, value, event = undefined) {
    const type = widget.type.includes("INT") ? Number : parseFloat
    widget.value = convertArrayToObject(value, Object.keys(value).length, type);
    //console.info(widget.value)
    if (
        widget.options &&
        widget.options.property &&
        node.properties[widget.options.property] !== undefined
        ) {
            node.setProperty(widget.options.property, widget.value)
        }
    if (widget.callback) {
        widget.callback(widget.value, app.canvas, node, pos, event)
    }
}

export function offsetDOMWidget(
    widget,
    ctx,
    node,
    widgetWidth,
    widgetY,
    height
  ) {
    const margin = 10
    const elRect = ctx.canvas.getBoundingClientRect()
    const transform = new DOMMatrix()
      .scaleSelf(
        elRect.width / ctx.canvas.width,
        elRect.height / ctx.canvas.height
      )
      .multiplySelf(ctx.getTransform())
      .translateSelf(0, widgetY + margin)

    const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
    Object.assign(widget.inputEl.style, {
      transformOrigin: '0 0',
      transform: scale,
      left: `${transform.e}px`,
      top: `${transform.d + transform.f}px`,
      width: `${widgetWidth}px`,
      height: `${(height || widget.parent?.inputHeight || 32) - margin}px`,
      position: 'absolute',
      background: !node.color ? '' : node.color,
      color: !node.color ? '' : 'white',
      zIndex: 5, //app.graph._nodes.indexOf(node),
    })
  }

export const hasWidgets = (node) => {
    if (!node.widgets || !node.widgets?.[Symbol.iterator]) {
      return false
    }
    return true
  }

export const cleanupNode = (node) => {
    if (!hasWidgets(node)) {
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

export function removeWidget(node, widgetOrSlot) {
    if (typeof widgetOrSlot === 'number') {
        node.widgets.splice(widgetOrSlot, 1);
    }
    else if (widgetOrSlot) {
        const index = node.widgets.indexOf(widgetOrSlot);
        if (index > -1) {
            node.widgets.splice(index, 1);
        }
    }
}

export function removeWidgets(who) {
    if (who.widgets) {
        for (const w of who.widgets) {
            w.onRemove();
        }
        who.widgets.length = 0;
    }
}

export function hideWidgetForGood(node, widget, suffix = '') {
    widget.origType = widget.type
    widget.origComputeSize = widget.computeSize
    widget.origSerializeValue = widget.serializeValue
    widget.computeSize = () => [0, -4]
    widget.type = CONVERTED_TYPE + suffix
    if (widget.linkedWidgets) {
      for (const w of widget.linkedWidgets) {
        hideWidgetForGood(node, w, ':' + widget.name)
      }
    }
  }

export function hideWidget(node, widget, suffix = '') {
    widget.origType = widget.type
    widget.hidden = true
    widget.origComputeSize = widget.computeSize
    widget.origSerializeValue = widget.serializeValue
    widget.computeSize = () => [0, -4]
    widget.type = CONVERTED_TYPE + suffix
    widget.serializeValue = () => {
        // Prevent serializing the widget if we have no input linked
        const { link } = node.inputs.find((i) => i.widget?.name === widget.name)
        if (link == null) {
            return undefined
        }
        return widget.origSerializeValue
            ? widget.origSerializeValue()
            : widget.value
    }

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
                hideWidget(node, w, ':' + widget.name)
        }
    }
}

  export function showWidget(widget) {
    widget.type = widget.origType
    widget.computeSize = widget.origComputeSize
    widget.serializeValue = widget.origSerializeValue

    delete widget.origType
    delete widget.origComputeSize
    delete widget.origSerializeValue

    // Hide any linked widgets, e.g. seed+seedControl
    if (widget.linkedWidgets) {
      for (const w of widget.linkedWidgets) {
        showWidget(w)
      }
    }
  }

  export function convertToWidget(node, widget) {
    showWidget(widget)
    const sz = node.size
    node.removeInput(node.inputs.findIndex((i) => i.widget?.name === widget.name))

    for (const widget of node.widgets) {
      widget.last_y -= LiteGraph.NODE_SLOT_HEIGHT
    }

    // Restore original size but grow if needed
    node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
  }

  /**
 * Extracts the type and link type from a widget config object.
 * @param {*} config
 * @returns
 */
export function getWidgetType(config) {
    // Special handling for COMBO so we restrict links based on the entries
    let type = config?.[0]
    let linkType = type
    if (type instanceof Array) {
      type = 'COMBO'
      linkType = linkType.join(',')
    }
    return { type, linkType }
  }

export function convertToInput(node, widget, config) {
    //console.log(node, widget)
    hideWidget(node, widget)

    const { linkType } = getWidgetType(config)

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

export const dynamic_connection = (node, index, connected, prefix='in_', type='*', names = []
    ) => {
        if (!node.inputs[index].name.startsWith(prefix)) {
            return
        }
        // remove all non connected inputs
        if (!connected && node.inputs.length > 1) {
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

export function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;')
    }
