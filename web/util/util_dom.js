/**
 * File: util_dom.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { convertArrayToObject } from './util.js'

export function renderTemplate(template, data) {
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g')
            template = template.replace(regex, data[key])
        }
    }
    return template
}

export function foldableToggle(elementId, symbolId) {
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

export function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;')
}
