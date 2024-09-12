/**
 * File: core_cozy_tips.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { $el } from "../../../scripts/ui.js"
import { setting_make } from '../util/util_api.js'
import { widgetGetHovered } from '../util/util_widget.js'

//const widget_height = 25;

const JTooltipWidget = (app, name, opts) => {
    let options = opts || {};
    options.serialize = false;
    const w = {
        name: name,
        type: "JTOOLTIP",
        hidden: true,
        options: options,
        draw: function () {
            return;
        },
        computeSize: function () {
            return [0, 0];
        }
    }
    return w
}

app.registerExtension({
    name: "jovimetrix.help.tooltips",
    async getCustomWidgets() {
        return {
            JTOOLTIP: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(JTooltipWidget(app, inputName, inputData[1]))
            })
        }
    },
    setup() {
        const tooltipEl = $el("div.jov-graph-tooltip", {
            parent: document.body,
        });

        let userTimeout = 750;
        let tooltipTimeout;
        let tips_previous;

        const hideTooltip = () => {
            if (tooltipTimeout) {
                clearTimeout(tooltipTimeout);
            }
            tooltipEl.style.display = "none";
            tips_previous = null;
        };

        const showTooltip = (tooltip, x_offset, y_offset) => {
            if (tooltipTimeout) {
                clearTimeout(tooltipTimeout);
            }
            if (tooltip && userTimeout > 0) {
                tooltipTimeout = setTimeout(() => {
                    tooltipEl.textContent = tooltip;
                    tooltipEl.style.display = "block";
                    tooltipEl.style.left = x_offset + "px";
                    tooltipEl.style.top = y_offset + "px";
                    const rect = tooltipEl.getBoundingClientRect();
                    if(rect.right > window.innerWidth) {
                        tooltipEl.style.left = (x_offset - rect.width) + "px";
                    }

                    if(rect.top < 0) {
                        tooltipEl.style.top = (y_offset + rect.height) + "px";
                    }
                }, userTimeout);
            }
        };

        const onCanvasPointerMove = function () {
            const node = this.node_over;

            if (!node || node.flags.collapsed) {
                return hideTooltip();
            }

            // Jovian tooltip logic
            const widget_tooltip = (node?.widgets || [])
                .find(widget => widget.type == 'JTOOLTIP');

            if (!widget_tooltip) {
                return hideTooltip();
            }

            const tips = widget_tooltip.options?.default;
            if (!tips) {
                return hideTooltip();
            }

            let tip;
            let name;
            //let mouse_x = app.canvas.mouse[0];
            //let mouse_y = app.canvas.mouse[1];

            const mouse_test_x = this.graph_mouse[0];
            const mouse_test_y = this.graph_mouse[1];
            const inputSlot = this.isOverNodeInput(node, mouse_test_x, mouse_test_y, [0, 0]);

            if (inputSlot !== -1) {
                const slot = node.inputs[inputSlot];
                tip = tips?.[slot.name];
                //mouse_y =  node.pos[1] - 5 * widget_height + inputSlot * widget_height;
                if (slot.widget) {
                    const widget = node.widgets.find(w => w.name == slot.name);
                    if (widget && widget.type.startsWith('converted-widget')) {
                        const def = widget.options?.default;
                        if (def) {
                            tip += ` (default: ${def})`;
                        }
                    }
                }
                name = `inputs_${inputSlot}`;
            } else {
                const outputSlot = this.isOverNodeOutput(node, mouse_test_x, mouse_test_y, [0, 0]);
                if (outputSlot !== -1) {
                    tip = tips?.['outputs']?.[outputSlot];
                    //mouse_y = node.pos[1] - 4 * widget_height + outputSlot * widget_height;
                    name = `outputs_${outputSlot}`;
                } else {
                    const hover = widgetGetHovered();
                    if (hover) {
                        const { widget } = hover;
                        if (widget && !widget.element) {
                            name = widget.name;
                            tip = tips?.[name];
                            //mouse_x = node.pos[0] - mouse_x + 10;
                            //mouse_y = node.pos[1] - 4 * widget_height;
                            const def = widget.options?.default;
                            if (def) {
                                tip += ` (default: ${def})`;
                            }
                        }
                    }
                }
            }
            if (tips_previous == name) {
                return;
            }

            tips_previous = name;
            if (!tip) {
                return hideTooltip();
            }
            showTooltip(tip, app.canvas.mouse[0], app.canvas.mouse[1] - 26);
        }.bind(app.canvas);

        const onChange = (val) => {
            if (val > 0) {
                LiteGraph.pointerListenerAdd(app.canvasEl, "move", onCanvasPointerMove);
            } else {
                LiteGraph.pointerListenerRemove(app.canvasEl, "move", onCanvasPointerMove);
            }
            userTimeout = val;
        }

        setting_make('tooltips 📝.delay', 'Delay', 'number',
            'How long (in milliseconds) to wait before showing the tooltip. 0 will turn it off.',
            50, {
                min: 0,
                step: 1,
            }, [], onChange);
    }
});