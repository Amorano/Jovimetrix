/**
 * File: widget_jlabel.js
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js"
// import * as util from '../util/util.js'
// import { offsetDOMWidget } from '../util/util_dom.js'

export const JLabelWidget = (label) => {
    const widget = {
        value: label,
        type: "JLABEL",
        options: {
            serialize: false,
        }
    };

    widget.draw = function(ctx, node, widget_width, y, widget_height) {

    }

    widget.mouse = function(event, pos, node) {

    },

    widget.computeSize = function() {
        return [0, 20];
    }

    return widget;
}

app.registerExtension({
    name: "jovimetrix.widget.jlabel",
    async getCustomWidgets(app) {
        return {
            JLABEL: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(JLabelWidget(app, inputName, inputData[0])),
            })
        }
    }
})
