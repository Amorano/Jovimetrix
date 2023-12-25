/**
 * File: widget_spinner.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'

export const JImageWidget = (app, inputName, inputData) => {
    const w = {
        name: inputName,
        type: inputData[0],
        value: val,
        draw: function (ctx, node, widgetWidth, widgetY, height) {
            const [cw, ch] = this.computeSize(widgetWidth)
            shared.offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, ch)
        },
        computeSize: function (width) {
            const ratio = this.inputRatio || 1
            if (width) {
                return [width, width / ratio + 4]
            }
            return [128, 128]
        },
        onRemoved: function () {
            if (this.inputEl) {
                this.inputEl.remove()
            }
        },
    }

    w.inputEl = document.createElement('img')
    w.inputEl.src = w.value
    w.inputEl.onload = function () {
        w.inputRatio = w.inputEl.naturalWidth / w.inputEl.naturalHeight
    }
    document.body.appendChild(w.inputEl)
    return w
}

const widgets = {
    name: "jovimetrix.widget.jimage",
    async getCustomWidgets(app) {
        return {
            JIMAGE: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(JImageWidget(app, inputName, inputData)),
            })
        }
    }
}
app.registerExtension(widgets)
