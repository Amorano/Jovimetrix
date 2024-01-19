/**
 * File: widget_vector.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'
import * as util_dom from '../core/util_dom.js'

export const VectorWidget = (app, inputName, options, initial, desc='') => {
    let isDragging;
    const offset = 4
    const label_width = 56
    const widget_padding = 16
    const widget_padding2 = 2 * widget_padding
    const label_full = widget_padding + label_width
    const values = options[1]?.default || initial;
    let val = {};
    for (let i = 0; i < values.length; i++) {
        val[i] = values[i];
    }

    const widget = {
        name: inputName,
        type: options[0],
        y: 0,
        value: val,
        options: options[1]
    }

    const precision = widget.options?.precision !== undefined ? widget.options.precision : 0;
    let step = options[0].includes('VEC') ? 0.01 : 1;
    widget.options.step = widget.options?.step || step;

    widget.draw = function(ctx, node, width, Y, height) {
        if (this.type !== options[0] && app.canvas.ds.scale > 0.5) return

        ctx.save()
        ctx.beginPath()
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR
        ctx.roundRect(widget_padding, Y, width - widget_padding2, height, 4)
        ctx.stroke()

        // label
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
        const label_center = (offset + label_full) / 2 - (inputName.length * 2.5)
        ctx.fillText(inputName, label_center, Y + height / 2 + offset)
        let x = label_full

        const fields = Object.keys(this?.value || [])
        const element_width = (width - label_full - widget_padding2) / fields.length

        for (const idx of fields) {
            ctx.save()
            ctx.beginPath()
            ctx.rect(x, Y, element_width, height)
            ctx.clip()
            ctx.fillStyle = LiteGraph.WIDGET_OUTLINE_COLOR
            ctx.fillRect(x - 1, Y, 2, height)
            ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
            const it = this.value[idx.toString()]
            const text = Number(it).toFixed(Math.min(2, precision)).toString()
            ctx.fillText(text, x + element_width / 2 - text.length * 2.5, Y + height / 2 + offset)
            ctx.restore()
            x += element_width
        }
        ctx.restore()
    }

    function clamp(w, v, idx) {
        if (w.options?.max !== undefined) {
            v = Math.min(v, w.options.max)
        }
        if (w.options?.min !== undefined) {
            v = Math.max(v, w.options.min)
        }
        w.value[idx] = (precision == 0) ? Number(v) : parseFloat(v).toFixed(precision)
    }

    widget.mouse = function (e, pos, node) {
        let delta = 0;

        if (e.type === 'pointerdown' && isDragging === undefined) {
            const x = pos[0] - label_full
            const size = Object.keys(this.value).length
            const element_width = (node.size[0] - label_full - widget_padding2) / size
            const index = Math.floor(x / element_width)
            if (index >= 0 && index < size) {
                isDragging = { name: this.name, idx: index}
            }
        }
        if (isDragging !== undefined && isDragging.idx > -1 && isDragging.name === this.name) {
            const idx = isDragging.idx
            const old_value = { ...this.value };
            if (e.type === 'pointermove' && e.deltaX) {
                let v = parseFloat(this.value[idx])
                v += this.options.step * Math.sign(e.deltaX)
                clamp(this, v, idx)
            } else if (e.type === 'pointerup') {
                if (e.click_time < 200 && delta == 0) {
                    const label = this.options?.label ? this.name + '➖' + this.options.label?.[idx] : this.name;
                    LGraphCanvas.active_canvas.prompt(label, this.value[idx], function(v) {
                        if (/^[0-9+\-*/()\s]+|\d+\.\d+$/.test(v)) {
                            try {
                                v = eval(v);
                            } catch (e) {}
                        }
                        if (this.value[idx] != v) {
                            setTimeout(
                                function () {
                                    clamp(this, v, idx)
                                    util_dom.inner_value_change(this, this.value, e)
                                }.bind(this), 20)
                        }
                    }.bind(this), e);
                }

                if (old_value != this.value) {
                    setTimeout(
                        function () {
                            //clamp(this, this.value[idx] || 0, idx)
                            util_dom.inner_value_change(this, this.value, e)
                        }.bind(this), 20)
                }
                isDragging = undefined
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
    name: "jovimetrix.widget.spinner",
    async getCustomWidgets(app) {
        return {
            VEC2: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(VectorWidget(app, inputName, inputData, [0, 0])),
            }),
            VEC3: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(VectorWidget(app, inputName, inputData, [0, 0, 0])),
            }),
            VEC4: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(VectorWidget(app, inputName, inputData, [0, 0, 0, 1])),
            })
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // export const myTypes = ['RGB', 'VEC2', 'VEC2', 'VEC3', 'VEC4', 'INT3', 'INT4', 'INT2']
        const myTypes = ['RGB', 'VEC2', 'VEC3', 'VEC4']
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
                    this.setSize?.(this.computeSize());
                    this.onRemoved = function () {
                        util.node_cleanup(this);
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
                        if (widget.type !== util.CONVERTED_TYPE && myTypes.includes(widget.type)) {
                            const who = matchingTypes.find(w => w[0] === widget.name)
                            const convertToInputObject = {
                                content: `Convert ${widget.name} to input`,
                                callback: () => util.convertToInput(this, widget, who[1])
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
        if (!nodeData.name.includes("(JOV)")) {
            return;
        }
    }
};

app.registerExtension(widgets)
