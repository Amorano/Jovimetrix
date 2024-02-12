/**
 * File: widget_rgb.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js"
import { TYPE_HIDDEN, fitHeight, node_cleanup, convertToInput } from '../core/util.js'

const PICKER_DEFAULT = '#ff0000'

export const RGBWidget = (key, val = PICKER_DEFAULT, compute = false) => {
    const widget = {
        name: key,
        type: 'RGB',
        value: val
    }
    widget.draw = function(ctx, node, widgetWidth, widgetY, height) {
        const hide = this.type !== 'RGB' && app.canvas.ds.scale > 0.5
        if (hide) return

        const border = 3
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR
        ctx.fillRect(0, widgetY, widgetWidth, height)
        ctx.fillStyle = this.value
        ctx.fillRect(border, widgetY + border, widgetWidth - border * 2, height - border * 2)

        const color = this.value.default || this.value
        if (!color) return
    }
    widget.mouse = function(e, pos, node) {
        if (e.type === 'pointerdown') {
            const widgets = node.widgets.filter((w) => w.type === 'COLOR')
            for (const w of widgets) {
                // color picker
            }
        }
    }
    widget.computeSize = function(width) {
        return [width, 32]
    }
    return widget
}

const widgets = {
    name: "jovimetrix.widgets",
    async getCustomWidgets(app) {
        return {
            RGB: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(RGBWidget(inputName, inputData[1]?.default || PICKER_DEFAULT)),
                minWidth: 35,
                minHeight: 35,
            })
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.name.includes("(JOV)")) {
            return;
        }
        const myTypes = ["RGB"]
        const inputTypes = nodeData.input;
        if (inputTypes) {
            const matchingTypes = ['required', 'optional']
                .flatMap(type => Object.entries(inputTypes[type] || [])
                    .filter(([_, value]) => myTypes.includes(value[0]))
                );

            // HAVE TO HOOK FOR CONNECTION CLEANUP ON REMOVE
            if (matchingTypes.length > 0) {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                    this.serialize_widgets = true;
                    fitHeight(this);
                    this.onRemoved = function () {
                        node_cleanup(this);
                    };
                    return r;
                };

                // MENU CONVERSIONS
                const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
                nodeType.prototype.getExtraMenuOptions = function (_, options) {
                    const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;
                    const convertToInputArray = [];
                    for (const w of matchingTypes) {
                        const widget = Object.values(this.widgets).find(m => m.name === w[0]);
                        if (widget.type !== TYPE_HIDDEN && myTypes.includes(widget.type)) {
                            const who = matchingTypes.find(w => w[0] === widget.name)
                            const convertToInputObject = {
                                content: `Convert ${widget.name} to input`,
                                callback: () => convertToInput(this, widget, who[1])
                            };
                            convertToInputArray.push(convertToInputObject);
                        }
                    }
                    const toInput = convertToInputArray;
                    if (toInput.length) {
                        options.push(...toInput, null);
                    }
                    return r;
                };
            }
        }
    }
};

app.registerExtension(widgets)
