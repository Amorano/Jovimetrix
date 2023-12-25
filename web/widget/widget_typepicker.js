/**
 * File: widget_typepicker.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'

export const TypePickerWidget = (app, inputName, inputData) => {
    const widget = {
        name: inputName,
        type: inputData[0],
        y: 0,
        value: val,
        options: inputData[1]
    }

    // widget.options.step = widget.options?.step || step;
    widget.computeSize = function (width) {
        return [width, LiteGraph.NODE_WIDGET_HEIGHT]
    }
    return widget
}

const widgets = {
    name: "jovimetrix.widget.typepicker",
    async getCustomWidgets(app) {
        return {
            TYPEPICKER: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(TypePickerWidget(app, inputName, inputData))
            })
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // for now filter out anyone trying to use this TYPE
        // this should dynamically create the "TYPE" per selection
        if (!nodeData.name.includes("(JOV)")) {
            return;
        }

        const myTypes = ["JTYPE"]
        const inputTypes = nodeData.input;
        if (inputTypes) {
            const matchingTypes = ['required', 'optional']
                .flatMap(type => Object.entries(inputTypes[type] || [])
                    .filter(([_, value]) => myTypes.includes(value[0]))
                );

            if (matchingTypes.length > 0) {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                    this.serialize_widgets = true;
                    /*
                    this.setSize?.(this.computeSize());
                    this.onRemoved = function () {
                        util.cleanupNode(this);
                    };
                    */
                    print(this)
                    return r;
                };
            }
        }
    }
};

app.registerExtension(widgets)
