/**
 * File: widget_jstring.js
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js"
import { offsetDOMWidget } from '../util/util_dom.js'

const withFont = (ctx, font, cb) => {
    const oldFont = ctx.font
    ctx.font = font
    cb()
    ctx.font = oldFont
}

const calculateTextDimensions = (ctx, value, width, fontSize = 12) => {
    const words = value.split(' ')
    const lines = []
    let currentLine = ''
    for (const word of words) {
      const testLine = currentLine.length === 0 ? word : `${currentLine} ${word}`
      const testWidth = ctx.measureText(testLine).width
      if (testWidth > width) {
        lines.push(currentLine)
        currentLine = word
      } else {
        currentLine = testLine
      }
    }
    if (lines.length === 0) lines.push(value)
    const textHeight = (lines.length + 1) * fontSize
    const maxLineWidth = lines.reduce(
      (maxWidth, line) => Math.max(maxWidth, ctx.measureText(line).width),
      0
    )
    return { textHeight, maxLineWidth }
  }

export const JStringWidget = (app, name, value) => {
    const fontSize = 16
    const w = {
        name: name,
        type: "JSTRING",
        value: value,
        draw: function (ctx, node, widgetWidth, widgetY, height) {
            offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, height)
        },
        computeSize(width) {
            if (!this.value) {
                return [32, 32]
            }
            if (!width) {
                console.error(`No width ${this.parent.size}`)
            }
            let dimensions
            withFont(app.ctx, `${fontSize}px monospace`, () => {
                dimensions = calculateTextDimensions(app.ctx, this.value, width)
            })
            const widgetWidth = Math.max(width || this.width || 32, dimensions.maxLineWidth)
            const widgetHeight = dimensions.textHeight * 1.5
            return [widgetWidth, widgetHeight]
        },
        onRemoved: function () {
            if (this.inputEl) {
                this.inputEl.remove()
            }
        },
        get value() {
            return this.inputEl.innerHTML
        },
        set value(val) {
            this.inputEl.innerHTML = val
            this.parent?.setSize?.(this.parent?.computeSize())
        },
    }

    w.inputEl = document.createElement('p')
    w.inputEl.style = `
        text-align: center;
        font-size: ${fontSize}px;
        color: var(--input-text);
        line-height: 0;
        font-family: monospace;
    `
    // w.value = val
    document.body.appendChild(w.inputEl)
    return w
}

app.registerExtension({
    name: "jovimetrix.widget.jstring",
    async getCustomWidgets(app) {
        return {
            JSTRING: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(JStringWidget(app, inputName, inputData[0])),
            })
        }
    }
})
