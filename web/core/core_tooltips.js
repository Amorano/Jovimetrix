/**
 * File: widget_tooltips.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js"

const ext = {
    name: "jovimetrix.tooltips",
    async getCustomWidgets(app) {
        return {
            JIMAGE: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(JImageWidget(app, inputName, inputData[0])),
            })
        }
    }
}
app.registerExtension(ext)
