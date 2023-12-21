/**
 * File: widget.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js"
import * as util from './util.js'

const _id = "jov.widgets.js"
const PICKER_DEFAULT = '#ff0000'

const RGBWidget = (key, val = PICKER_DEFAULT, compute = false) => {
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

const SpinnerWidget = (app, inputName, inputData, initial, desc='') => {
    const offset = 4
    const label_width = 50
    const widget_padding = 15
    const widget_padding2 = 2 * widget_padding
    const label_full = widget_padding + label_width

    let regions = []
    let isDragging

    console.info(inputData)

    const widget = {
        name: inputName,
        type: inputData[0],
        y: 0,
        value: inputData?.default || initial,
        options: inputData[1]
    }

    widget.draw = function(ctx, node, width, Y, height) {
        if (this.type !== inputData[0] && app.canvas.ds.scale > 0.5) return
        const element_width = (width - label_full - widget_padding2) / this.value.length

        ctx.save()
        ctx.beginPath()
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR
        ctx.roundRect(widget_padding, Y, width - widget_padding2, height, 8)
        ctx.stroke()

        // label
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
        const label_center = (offset + label_full) / 2 - (inputName.length * 1.5)
        ctx.fillText(inputName, label_center, Y + height / 2 + offset)
        let x = label_full
        regions = []
        for (let idx = 0; idx < this.value.length; idx++) {
            ctx.save()
            ctx.beginPath()
            ctx.rect(x, Y, element_width, height)
            ctx.clip()
            ctx.fillStyle = LiteGraph.WIDGET_OUTLINE_COLOR
            ctx.fillRect(x - 1, Y, 2, height)
            ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR

            // value
            const precision = this.options?.precision !== undefined ? this.options.precision : 0;
            const text = Number(this.value[idx]).toFixed(precision).toString()
            ctx.fillText(text, x + element_width / 2 - text.length * 1.5, Y + height / 2 + offset)
            ctx.restore()
            regions.push([x + 1 - element_width, x - 1])
            x += element_width
        }
        ctx.restore()
    }

    widget.mouse = function (e, pos, node) {
        let delta = 0;
        if (e.type === 'pointerdown' && isDragging === undefined) {
            //console.info(node)
            const x = pos[0] - label_full
            const element_width = (node.size[0] - label_full - widget_padding2) / regions.length
            const index = Math.floor(x / element_width)
            if (index >= 0 && index < regions.length) {
                isDragging = { name: this.name, idx: index}
            }
        }
        else if (isDragging !== undefined && isDragging.idx > -1 && isDragging.name === this.name) {
            const idx = isDragging.idx;
            let old_value = this.value.slice();
            if (e.type === 'pointermove') {
                let v = this.value[idx];
                if (e.deltaX) {
                    delta = e.deltaX
                    v += ((this.options?.step || 8) * Math.sign(delta))
                    if (this.options?.max !== undefined) {
                        v = Math.min(v, this.options.max)
                    }
                    if (this.options?.min !== undefined) {
                        v = Math.max(v, this.options.min)
                    }
                    this.value[idx] = v;
                }
            } else if (e.type === 'pointerup') {
                isDragging = undefined
                if (e.click_time < 200 && delta == 0) {
                    const label = this.options?.label ? this.name + '➖' + this.options.label?.[idx] : this.name;
                    LGraphCanvas.active_canvas.prompt(label, this.value[idx], function(v) {
                        // check if v is a valid equation or a number
                        if (/^[0-9+\-*/()\s]+|\d+\.\d+$/.test(v)) {
                            try {
                                v = eval(v);
                            } catch (e) {}
                        }
                        v = parseFloat(v) || 0;
                        if (this.options?.max !== undefined) {
                            v = Math.min(v, this.options.max)
                        }
                        if (this.options?.min !== undefined) {
                            v = Math.max(v, this.options.min)
                        }
                        if (old_value[idx] != v)
                        {
                            this.value[idx] = v;
                            util.inner_value_change(this, this.value, e);
                            console.info(node)
                        }
                    }.bind(this), e);
                }
                else if (old_value[idx] != this.value[idx])
                {
                    //util.inner_value_change(this, this.value, e);
                    console.info(node.onWidgetChanged)
                    /*
                    old_value[idx] = this.value[idx]
                    setTimeout(
                        function () {
                            //console.info(this, this.value)
                            util.inner_value_change(this, this.value, e)
                        }.bind(this), 20
                    )*/
                }
                app.canvas.setDirty(true)
            }
        }
    }

    widget.computeSize = function (width) {
        return [width, LiteGraph.NODE_WIDGET_HEIGHT]
    }

    widget.desc = desc
    return widget
}

const widgets = {
    name: _id,
    async getCustomWidgets(app) {
        return {
            RGB: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(RGBWidget(inputName, inputData[1]?.default || PICKER_DEFAULT)),
                minWidth: 35,
                minHeight: 35,
            }),
            INTEGER2: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, inputName, inputData, [0, 0], 'Represents a Bounding Box with x, y, width, and height.')),
            }),
            FLOAT2: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, inputName, inputData, [0., 0.])),
            }),
            INTEGER3: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, inputName, inputData, [0, 0, 0])),
            }),
            FLOAT3: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, inputName, inputData, [0., 0., 0.])),
            }),
            INTEGER4: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, inputName, inputData, [0, 0, 0, 255])),
            }),
            FLOAT4: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, inputName, inputData, [0., 0., 0., 1.])),
            }),
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const inputTypes = nodeData.input;
        if (inputTypes) {
            const matchingTypes = ['required', 'optional']
                .flatMap(type => Object.entries(inputTypes[type] || [])
                    .filter(([_, value]) => util.newTypes.includes(value[0]))
                );

            // HAVE TO HOOK FOR CONNECTION CLEANUP ON REMOVE
            if (matchingTypes.length > 0) {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                    this.serialize_widgets = true;
                    this.setSize?.(this.computeSize());
                    this.onRemoved = function () {
                        util.cleanupNode(this);
                    };
                    return r;
                };

                // MENU CONVERSIONS
                const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
                nodeType.prototype.getExtraMenuOptions = function (_, options) {
                    const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;
                    const toInput = matchingTypes
                        .map(w => Object.values(this.widgets).find(m => m.name === w[0]))
                        .filter(widget => widget.type !== util.CONVERTED_TYPE && util.newTypes.includes(widget.type))
                        .map(widget => ({
                            content: `Convert ${widget.name} to input`,
                            callback: () => util.convertToInput(this, widget, matchingTypes.find(w => w[0] === widget.name))
                        }));

                    if (toInput.length) {
                        options.push(...toInput, null);
                    }
                    return r;
                };
            }
        }
        if (!nodeData.name.includes("(JOV)")) {
            return;
        }
    }
};

app.registerExtension(widgets)
