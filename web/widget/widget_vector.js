/**
 * File: widget_vector.js
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js"
import { widgetToInput, widgetToWidget } from '../util/util_widget.js'
import { domInnerValueChange, colorHex2RGB, colorRGB2Hex } from '../util/util.js'
import { $el } from "../../../scripts/ui.js"

const VectorWidget = (app, inputName, options, initial, desc='') => {
    const values = options[1]?.default || initial;
    const widget = {
        name: inputName,
        type: options[0],
        y: 0,
        value: values,
        options: options[1]
    }

    if (options[0].endsWith('INT')) {
        widget.options.step = 1;
        widget.options.round = 1;
        widget.options.precision = 0;
        widget.options.step = 1;
        if (widget.options?.rgb || false) {
            widget.options.max = 255;
            widget.options.min = 0;
            // add the label for being an RGB(A) field?
            // "label": [Lexicon.R, Lexicon.G, Lexicon.B, Lexicon.A],
        }
    } else {
        widget.options.precision = widget.options?.precision || 6;
        widget.options.step = widget.options?.step || 0.001;
        widget.options.round = widget.options?.round || 1 / 10 ** widget.options.step;
    }

    const offset_y = 4;
    const widget_padding_left = 13;
    const widget_padding = 30;
    const label_full = 72;
    const label_center = label_full/2;
    let picker;
    let isDragging;

    widget.draw = function(ctx, node, width, Y, height) {
        if ((app.canvas.ds.scale < 0.50) || (!this.type.startsWith("VEC") && this.type != "COORD2D")) return;
        ctx.save()
        ctx.beginPath()
        ctx.lineWidth = 1
        ctx.fillStyle = LiteGraph.WIDGET_OUTLINE_COLOR
        ctx.roundRect(widget_padding_left+2, Y, width - widget_padding, height, 15)
        ctx.stroke()
        ctx.lineWidth = 1
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR
        ctx.roundRect(widget_padding_left+2, Y, width - widget_padding, height, 15)
        ctx.fill()

        // label
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
        ctx.fillText(inputName, label_center - (inputName.length * 1.5), Y + height / 2 + offset_y)
        let x = label_full + 1

        const fields = Object.keys(this?.value || []);
        let count = fields.length;
        if (widget.options?.rgb) {
            count += 0.23;
        }
        const element_width = (width - label_full - widget_padding) / count;
        const element_width2 = element_width / 2;

        let converted = [];
        for (const idx of fields) {
            ctx.save()
            ctx.beginPath()
            ctx.fillStyle = LiteGraph.WIDGET_OUTLINE_COLOR
            // separation bar
            if (idx != fields.length || (idx == fields.length && !this.options.rgb)) {
                ctx.moveTo(x, Y)
                ctx.lineTo(x, Y+height)
                ctx.stroke();
            }

            // value
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR
            const it = this.value[idx.toString()];
            const precision = widget.options?.precision || 0;
            let value = Number(it);
            if (precision > 0) {
                value = value.toFixed(Math.min(2, precision));
            }
            converted.push(value);
            const text = value.toString();
            ctx.fillText(text, x + element_width2 - text.length * 3.3, Y + height/2 + offset_y);
            ctx.restore();
            x += element_width;
        }

        if (this.options?.rgb) {
            try {
                ctx.fillStyle = colorRGB2Hex(converted);
            } catch (e) {
                console.error(e)
                ctx.fillStyle = "#FFF";
            }
            ctx.roundRect(width - 1.17 * widget_padding, Y+1, 19, height-2, 16);
            ctx.fill()
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
        const precision = widget.options?.precision !== undefined ? widget.options.precision : 0;
        w.value[idx] = (precision == 0) ? Number(v) : parseFloat(v).toFixed(precision)
    }

    widget.mouse = function (e, pos, node) {
        let delta = 0;
        if (e.type === 'pointerdown' & isDragging === undefined) {
            const x = pos[0] - label_full;
            const size = Object.keys(this.value).length;;
            const element_width = (node.size[0] - label_full - widget_padding * 1.25) / size;
            const index = Math.floor(x / element_width);
            if (index >= 0 && index < size) {
                isDragging = { name: this.name, idx: index}
            } else if (this.options.rgb) {
                const rgba = Object.values(this?.value || []);
                let color = colorRGB2Hex(rgba.slice(0, 3));
                if (index == size) {
                    if (!picker) {
                        picker = $el("input", {
                            type: "color",
                            parent: document.body,
                            style: {
                                display: "none",
                            },
                        });
                        picker.onchange = () => {
                            if (picker.value) {
                                this.value = colorHex2RGB(picker.value);
                                if (rgba.length > 3) {
                                    this.value.push(rgba[3])
                                }
                            }
                        };
                    }
                    picker.value = color;
                    picker.click();
                } else if (x < 0 && rgba.length > 2) {
                    const target = Object.values(rgba.map(item => 255 - item)).slice(0, 3);
                    this.value = Object.values(this.value);
                    this.value.splice(0, 3, ...target);
                }
            }
        }

        if (isDragging !== undefined && isDragging.idx > -1 && isDragging.name === this.name) {
            const idx = isDragging.idx
            const old_value = { ...this.value };
            if (e.type === 'pointermove' && e.deltaX) {
                let v = parseFloat(this.value[idx]);
                v += this.options.step * Math.sign(e.deltaX);
                clamp(this, v, idx);
            } else if (e.type === 'pointerup') {
                isDragging = undefined
                if (e.click_time < 150 && delta == 0) {
                    const label = this.options?.label ? this.name + '➖' + this.options.label?.[idx] : this.name;
                    LGraphCanvas.active_canvas.prompt(label, this.value[idx], function(v) {
                        if (/^[0-9+\-*/()\s]+|\d+\.\d+$/.test(v)) {
                            try {
                                v = eval(v);
                            } catch (e) {

                            }
                        }
                        if (this.value[idx] != v) {
                            setTimeout(
                                function () {
                                    clamp(this, v, idx)
                                    domInnerValueChange(node, pos, this, this.value, e)
                                }.bind(this), 20)
                        }
                    }.bind(this), e);
                }

                if (old_value != this.value) {
                    setTimeout(
                        function () {
                            domInnerValueChange(node, pos, this, this.value, e)
                        }.bind(this), 20)
                }
            }
        }
        app.canvas.setDirty(true, true);
    }

    widget.serializeValue = async () => {
        if (widget.value === null) {
            return null;
        }
        let value = widget.value;
        if (Array.isArray(widget.value)) {
            value = widget.value.reduce((acc, tuple, index) => ({ ...acc, [index]: tuple }), {});
        }

        return value;
    }

    widget.desc = desc
    return widget
}

app.registerExtension({
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
                widget: node.addCustomWidget(VectorWidget(app, inputName, inputData, [0, 0, 0, 0])),
            }),
            VEC2INT: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(VectorWidget(app, inputName, inputData, [0, 0])),
            }),
            VEC3INT: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(VectorWidget(app, inputName, inputData, [0, 0, 0])),
            }),
            VEC4INT: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(VectorWidget(app, inputName, inputData, [0, 0, 0, 0])),
            })
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const myTypes = ['RGB', 'VEC2', 'VEC3', 'VEC4', 'VEC2INT', 'VEC3INT', 'VEC4INT']
        const inputTypes = nodeData.input;
        if (inputTypes) {
            const matchingTypes = ['required', 'optional']
                .flatMap(type => Object.entries(inputTypes[type] || [])
                    .filter(([_, value]) => myTypes.includes(value[0]))
                );

            // CLEANUP ON REMOVE
            if (matchingTypes.length > 0) {
                // MENU CONVERSIONS
                const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
                nodeType.prototype.getExtraMenuOptions = function (_, options) {
                    const me = getExtraMenuOptions?.apply(this, arguments);
                    const widgetToInputArray = [];
                    for (const [widgetName, additionalInfo] of matchingTypes) {
                        const widget = Object.values(this.widgets).find(m => m.name === widgetName);
                        if (myTypes.includes(widget.type) || widget.type.endsWith('-jov')) {
                            if (!widget.hidden) {
                                const widgetToInputObject = {
                                    content: `Convert ${widgetName} to input`,
                                    callback: () => widgetToInput(this, widget, additionalInfo)
                                };
                                widgetToInputArray.push(widgetToInputObject);
                            } else {
                                const widgetToInputObject = {
                                    content: `Convert ${widgetName} to widget`,
                                    callback: () => widgetToWidget(this, widget, additionalInfo)
                                };
                                widgetToInputArray.push(widgetToInputObject);
                            }
                        }
                    }
                    if (widgetToInputArray.length) {
                        options.push(...widgetToInputArray, null);
                    }
                    return me;
                };
            }
        }
    }
})
