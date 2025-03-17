/**/

import { app } from "../../../scripts/app.js"
import { $el } from "../../../scripts/ui.js"
import { widgetToInput, widgetToWidget } from '../util/util_widget.js'
import { domInnerValueChange } from '../util/util.js'
/** @import { IWidget, LGraphCanvas } from '../../types/litegraph/litegraph.d.ts' */

const GET_CONFIG = Symbol();
const TYPES = ['RGB', 'VEC2', 'VEC3', 'VEC4', 'VEC2INT', 'VEC3INT', 'VEC4INT']
const TYPES_ACCEPT = 'RGB, VEC2, VEC3, VEC4, VEC2INT, VEC3INT, VEC4INT, FLOAT, INT, BOOLEAN'

function convertToInput(node, widget) {
    // hideWidget(node, widget);
    //const { type } = getWidgetType(config);
    // const [oldWidth, oldHeight] = node.size;
    /*
    const config = [
        widget.type,
        widget.options || {}
    ];
    */
    //const inputIsOptional = !!widget.options?.inputIsOptional;
    //widget.options.type = TYPES_ACCEPT;
    const input = node.addInput(widget.name, widget.type, {
        // @ts-expect-error [GET_CONFIG] is not a valid property of IWidget
        widget: { name: widget.name, [GET_CONFIG]: () => [
            widget.type,
            widget.options || {}
        ] },
        //...inputIsOptional ? { shape: LiteGraph.SlotShape.HollowCircle } : {}
        ...{ shape: LiteGraph.SlotShape.HollowCircle }
    });
    /*
    for (const widget2 of node.widgets) {
        widget2.last_y += LiteGraph.NODE_SLOT_HEIGHT;
    }
    node.setSize([
        Math.max(oldWidth, node.size[0]),
        Math.max(oldHeight, node.size[1])
    ]);
    */
    return input;
}

function isVersionLess(v1, v2) {
    const parts1 = v1.split('.').map(Number);
    const parts2 = v2.split('.').map(Number);

    for (let i = 0; i < Math.max(parts1.length, parts2.length); i++) {
        const num1 = parts1[i] || 0;
        const num2 = parts2[i] || 0;
        if (num1 < num2) return true;
        if (num1 > num2) return false;
    }
    return false; // They are equal
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

const VectorWidget = (app, inputName, options, initial, desc='') => {
    const values = options[1]?.default || initial;
    /** @type {IWidget} */
    const widget = {
        name: inputName,
        type: options[0],
        y: 0,
        value: values,
        options: options[1]
    }

    if (widget.options?.rgb || false) {
        widget.options.maj = 255;
        widget.options.mij = 0;
        widget.options.label = ['🟥', '🟩', '🟦', 'ALPHA'];
    }

    widget.options.precision = 3;
    widget.options.step = 0.0075;
    widget.options.round = 1 / 10 ** widget.options.step;

    if (options[0].endsWith('INT')) {
        widget.options.step = 1;
        widget.options.round = 1;
        widget.options.precision = 0;
        widget.options.step = 1;
    } else {
        if (widget.options?.rgb || false) {
            widget.options.maj = 1;
        }
    }

    const offset_y = 4;
    const widget_padding_left = 13;
    const widget_padding = 30;
    const label_full = 72;
    const label_center = label_full/2;
    /** @type {HTMLInputElement} */
    let picker;

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
            ctx.roundRect(width - 1.17 * widget_padding, Y+1, 19, height-2, 16);
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
                            // Suppressed exception
                        }
                    }
                    if (this.value[index] != v) {
                        setTimeout(
                            function () {
                                clamp(this, v, index);
                                domInnerValueChange(node, pos, this, this.value, eUp);
                            }.bind(this), 20)
                    }
                }.bind(this), eUp);

                if (old_value != this.value) {
                    setTimeout(
                        function () {
                            domInnerValueChange(node, pos, this, this.value, eUp);
                        }.bind(this), 20);
                }

                return
            }
            if (!this.options?.rgb) return;

            //const rgba = widget.value;
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

            let v = parseFloat(this.value[index]);
            v += this.options.step * Math.sign(eMove.deltaX);
            clamp(this, v, index);
        }
    }

    widget.serializeValue = async () => {
        const value = widget.value;
        if (value === null) {
            return null;
        }

        if (Array.isArray(value)) {
            return value.reduce((acc, tuple, index) => ({ ...acc, [index]: tuple }), {});
        }
        return value;
    }

    widget.desc = desc;
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
        const inputTypes = nodeData.input;
        if (inputTypes.length == 0) {
            return;
        }

        const matchingTypes = ['required', 'optional']
        .flatMap(type => Object.entries(inputTypes[type] || [])
            .filter(([_, value]) => TYPES.includes(value[0]))
        );
        if (matchingTypes.length == 0) {
            return;
        }

        const version = window.__COMFYUI_FRONTEND_VERSION__;
        if (!isVersionLess(version, "1.10.3")) {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                const me = onNodeCreated?.apply(this);
                Object.entries(this.widgets).forEach(([key, widget]) => {
                    if (!TYPES.includes(widget.type)) {
                        return;
                    }
                    convertToInput(this, widget);
                });
                return me;
            }
            return;
        }

        // MENU CONVERSIONS
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const me = getExtraMenuOptions?.apply(this, arguments);
            const widgetToInputArray = [];
            for (const [widgetName, additionalInfo] of matchingTypes) {
                const widget = Object.values(this.widgets).find(m => m.name == widgetName);
                if (TYPES.includes(widget.type) || widget.type.endsWith('-jov')) {
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
})
