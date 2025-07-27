/**/

import { app } from "../../scripts/app.js"
import { $el } from "../../scripts/ui.js"
/** @import { IWidget, LGraphCanvas } from '../../types/litegraph/litegraph.d.ts' */

function arrayToObject(values, length, parseFn) {
    const result = {};
    for (let i = 0; i < length; i++) {
        result[i] = parseFn(values[i]);
    }
    return result;
}

function domInnerValueChange(node, pos, widget, value, event=undefined) {
    //const numtype = widget.type.includes("INT") ? Number : parseFloat
    widget.value = arrayToObject(value, Object.keys(value).length, widget.convert);
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

function colorHex2RGB(hex) {
    hex = hex.replace(/^#/, '');
    const bigint = parseInt(hex, 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return [r, g, b];
}

function colorRGB2Hex(input) {
    const rgbArray = typeof input == 'string' ? input.match(/\d+/g) : input;
    if (rgbArray.length < 3) {
        throw new Error('input not 3 or 4 values');
    }
    const hexValues = rgbArray.map((value, index) => {
        if (index == 3 && !value) return 'ff';
        const hex = parseInt(value).toString(16);
        return hex.length == 1 ? '0' + hex : hex;
    });
    return '#' + hexValues.slice(0, 3).join('') + (hexValues[3] || '');
}

const VectorWidget = (app, inputName, options, initial) => {
    const values = options[1]?.default || initial;
    /** @type {IWidget} */
    const widget = {
        name: inputName,
        type: options[0],
        y: 0,
        value: values,
        options: options[1]
    }

    widget.convert = parseFloat;
    widget.options.precision = widget.options?.precision || 2;
    widget.options.step = widget.options?.step || 0.1;
    widget.options.round = 1 / 10 ** widget.options.step;

    if (widget.options?.rgb || widget.options?.int || false) {
        widget.options.step = 1;
        widget.options.round = 1;
        widget.options.precision = 0;
        widget.convert = Number;
    }

    if (widget.options?.rgb || false) {
        widget.options.maj = 255;
        widget.options.mij = 0;
        widget.options.label = ['🟥', '🟩', '🟦', 'ALPHA'];
    }

    const offset_y = 4;
    const widget_padding_left = 13;
    const widget_padding = 30;
    const label_full = 72;
    const label_center = label_full / 2;

    /** @type {HTMLInputElement} */
    let picker;

    widget.draw = function(ctx, node, width, Y, height) {
        // if ((app.canvas.ds.scale < 0.50) || (!this.type2.startsWith("VEC") && this.type2 != "COORD2D")) return;
        if ((app.canvas.ds.scale < 0.50) || (!this.type.startsWith("VEC"))) return;
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
            if (idx != fields.length || (idx == fields.length && !this.options?.rgb)) {
                ctx.moveTo(x, Y)
                ctx.lineTo(x, Y+height)
                ctx.stroke();
            }

            // value
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR
            const it = this.value[idx.toString()];
            let value = (widget.options.precision == 0) ? Number(it) : parseFloat(it).toFixed(widget.options.precision);
            converted.push(value);
            const text = value.toString();
            ctx.fillText(text, x + element_width2 - text.length * 3.3, Y + height/2 + offset_y);
            ctx.restore();
            x += element_width;
        }

        if (this.options?.rgb && converted.length > 2) {
            try {
                ctx.fillStyle = colorRGB2Hex(converted);
            } catch (e) {
                console.error(converted, e);
                ctx.fillStyle = "#FFF";
            }
            ctx.roundRect(width-1.17 * widget_padding, Y+1, 19, height-2, 16);
            ctx.fill()
        }
        ctx.restore()
    }

    function clamp(widget, v, idx) {
        v = Math.min(v, widget.options?.maj !== undefined ? widget.options.maj : v);
        v = Math.max(v, widget.options?.mij !== undefined ? widget.options.mij : v);
        widget.value[idx] = (widget.options.precision == 0) ? Number(v) : parseFloat(v).toFixed(widget.options.precision);
    }

    /**
     * @todo ▶️, 🖱️, 😀
     * @this IWidget
     */
    widget.onPointerDown = function (pointer, node, canvas) {
        const e = pointer.eDown
        const x = e.canvasX - node.pos[0] - label_full;
        const size = Object.keys(this.value).length;
        const element_width = (node.size[0] - label_full - widget_padding * 1.25) / size;
        const index = Math.floor(x / element_width);

        pointer.onClick = (eUp) => {
            /* if click on header, reset to defaults */
            if (index == -1 && eUp.shiftKey) {
                widget.value = Object.assign({}, widget.options.default);
                return;
            }
            else if (index >= 0 && index < size) {
                const pos = [eUp.canvasX - node.pos[0], eUp.canvasY - node.pos[1]]
                const old_value = { ...this.value };
                const label = this.options?.label ? this.name + '➖' + this.options.label?.[index] : this.name;

                LGraphCanvas.active_canvas.prompt(label, this.value[index], function(v) {
                    if (/^[0-9+\-*/()\s]+|\d+\.\d+$/.test(v)) {
                        try {
                            v = eval(v);
                        } catch {
                            v = old_value[index];
                        }
                    } else {
                        v = old_value[index];
                    }

                    if (this.value[index] != v) {
                        setTimeout(
                            function () {
                                clamp(this, v, index);
                                domInnerValueChange(node, pos, this, this.value, eUp);
                            }.bind(this), 5)
                    }
                }.bind(this), eUp);
                return;
            }
            if (!this.options?.rgb) return;

            const rgba = Object.values(this?.value || []);
            const color = colorRGB2Hex(rgba.slice(0, 3));

            if (index != size && (x < 0 && rgba.length > 2)) {
                const target = Object.values(rgba.map((item) => 255 - item)).slice(0, 3);
                this.value = Object.values(this.value);
                this.value.splice(0, 3, ...target);
                return
            }

            if (!picker) {
                // firefox?
                //position: "absolute", // Use absolute positioning for consistency
                //left: `${eUp.pageX}px`, // Use pageX for more consistent placement
                //top: `${eUp.pageY}px`,
                picker = $el("input", {
                    type: "color",
                    parent: document.body,
                    style: {
                        position: "fixed",
                        left: `${eUp.clientX}px`,
                        top: `${eUp.clientY}px`,
                        height: "0px",
                        width: "0px",
                        padding: "0px",
                        opacity: 0,
                    },
                });
                picker.addEventListener('blur', () => picker.style.display = 'none')
                picker.addEventListener('input', () => {
                    if (!picker.value) return;

                    widget.value = colorHex2RGB(picker.value);
                    if (rgba.length > 3) {
                        widget.value.push(rgba[3]);
                    }
                    canvas.setDirty(true)
                })
            } else {
                picker.style.display = 'revert'
                picker.style.left = `${eUp.clientX}px`
                picker.style.top = `${eUp.clientY}px`
            }
            picker.value = color;
            requestAnimationFrame(() => {
                picker.showPicker()
                picker.focus()
            })
        }

        pointer.onDrag = (eMove) => {
            if (!eMove.deltaX || !(index > -1)) return;
            if (index >= size) return;
            let v = parseFloat(this.value[index]);
            v += this.options.step * Math.sign(eMove.deltaX);
            clamp(this, v, index);
            if (widget.callback) {
                widget.callback(widget.value, app.canvas, node)
            }
        }
    }

    widget.serializeValue = async (node, index) => {
        const rawValues = Array.isArray(widget.value)
            ? widget.value
            : Object.values(widget.value);
        const funct = widget.options?.int ? Number : parseFloat;
        return rawValues.map(v => funct(v));
    };

    return widget;
}

app.registerExtension({
    name: "jovi.widget.spinner",
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
            })
        }
    }
})
