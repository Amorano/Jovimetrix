/**
 * File: util.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { api } from "/scripts/api.js"

const CONVERTED_TYPE = "converted-widget";
export const newTypes = ['RGB', 'FLOAT2', 'FLOAT3', 'FLOAT4', 'INTEGER2', 'INTEGER3', 'INTEGER4']

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

export let NODE_LIST = await api_get("./../object_info")
export let CONFIG_CORE = await api_get("/jovimetrix/config")
export let CONFIG_USER = CONFIG_CORE.user.default
export let CONFIG_COLOR = CONFIG_USER.color
export let THEME = CONFIG_COLOR.theme
export let USER = 'user.default'

// gets the CONFIG entry for name
export function node_color_get(find_me) {
    let node = THEME[find_me]
    if (node) {
        return node
    }
    node = NODE_LIST[find_me]
    //console.info(node)
    if (node && node.category) {
        //console.info(CONFIG)
        const segments = node.category.split('/')
        let k = segments.join('/')
        while (k) {
            const found = THEME[k]
            if (found) {
                //console.info(found, node.category)
                return found
            }
            const last = k.lastIndexOf('/')
            k = last !== -1 ? k.substring(0, last) : ''
        }
    }
}

// refresh the color of a node
export function node_color_reset(node, refresh=true) {
    const data = node_color_get(node.type || node.name)
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
    return "#13171DFF"
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

// 100% stolen from MTB
// https://github.com/melMass/comfy_mtb/blob/main/web/comfy_shared.js
export function inner_value_change(widget, value, event = undefined) {

    switch (widget.type) {
        case "INTEGER2":
            value = [Number(value[0]), Number(value[1])]
            break
        case "INTEGER3":
            value = [Number(value[0]), Number(value[1]), Number(value[2])]
            break
        case "INTEGER4":
            value = [Number(value[0]), Number(value[1]), Number(value[2]), Number(value[3])]
            break
        case "FLOAT2":
            value = [Number(value[0]), Number(value[1])]
            break
        case "FLOAT3":
            value = [Number(value[0]), Number(value[1]), Number(value[2])]
            break
        case "FLOAT4":
            value = [Number(value[0]), Number(value[1]), Number(value[2]), Number(value[3])]
            break
    }
    widget.value = value
    if (
        widget.options &&
        widget.options.property &&
        node.properties[widget.options.property] !== undefined
        ) {
            node.setProperty(widget.options.property, value)
        }
    if (widget.callback) {
        widget.callback(widget.value, app.canvas, node, pos, event)
    }
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

export function hideWidget(node, widget, suffix = '') {
    widget.origType = widget.type
    widget.hidden = true
    widget.origComputeSize = widget.computeSize
    widget.origSerializeValue = widget.serializeValue
    widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
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
    console.log(node, widget)
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