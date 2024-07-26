/**
 * File: core_cozy_tips.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { $el } from "../../../scripts/ui.js"
import { getHoveredWidget } from '../util/util_widget.js'

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
    async getCustomWidgets(app) {
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
        const hideTooltip = () => {
            if (tooltipTimeout) {
                clearTimeout(tooltipTimeout);
            }
            tooltipEl.style.display = "none";
        };

        const showTooltip = (tooltip) => {
            if (tooltipTimeout) {
                clearTimeout(tooltipTimeout);
            }
            if (tooltip && userTimeout > 0) {
                tooltipTimeout = setTimeout(() => {
                    tooltipEl.textContent = tooltip;
                    tooltipEl.style.display = "block";
                    tooltipEl.style.left = app.canvas.mouse[0] + "px";
                    tooltipEl.style.top = app.canvas.mouse[1] + "px";
                    const rect = tooltipEl.getBoundingClientRect();
                    if(rect.right > window.innerWidth) {
                        tooltipEl.style.left = (app.canvas.mouse[0] - rect.width) + "px";
                    }

                    if(rect.top < 0) {
                        tooltipEl.style.top = (app.canvas.mouse[1] + rect.height) + "px";
                    }
                }, userTimeout);
            }
        };

        const onCanvasPointerMove = function () {
            hideTooltip();
            const node = this.node_over;
            if (!node) return;

            if (node.flags.collapsed) return;

            // jovian tooltip
            const widget_tooltip = (node?.widgets || [])
                .find(widget => widget.type === 'JTOOLTIP');
            if (!widget_tooltip) return;
            const tips = widget_tooltip.options.default || {};

            const inputSlot = this.isOverNodeInput(node, this.graph_mouse[0], this.graph_mouse[1], [0, 0]);
            if (inputSlot !== -1) {
                const slot = node.inputs[inputSlot];
                let tip = tips?.[slot.name];
                if (slot.widget) {
                    const widget = node.widgets.find(w => w.name === slot.name);
                    if (widget && widget.type.startsWith('converted-widget')) {
                        const def = widget.options?.default;
                        if (def) {
                            tip += ` (default: ${def})`;
                        }
                    }
                }
                return showTooltip(tip);
            }

            const outputSlot = this.isOverNodeOutput(node, this.graph_mouse[0], this.graph_mouse[1], [0, 0]);
            if (outputSlot !== -1) {
                let tip = tips?.['outputs']?.[outputSlot];
                return showTooltip(tip);
            }

            const widget = getHoveredWidget();
            if (widget && !widget.element) {
                let tip = tips?.[widget.name];
                const def = widget.options?.default;
                if (def) {
                    tip += ` (default: ${def})`;
                }
                return showTooltip(tip);
            }
        }.bind(app.canvas);

        app.ui.settings.addSetting({
            id: "jovimetrix.cozy.tips",
            name: "🇯 🎨 Tooltips Delay",
            tooltip: "How long (in milliseconds) to wait before showing the tooltip. 0 will turn it off.",
            type: "number",
            defaultValue: 500,
            attrs: {
                min: 0,
                step: 1,
            },
            onChange(value) {
                if (value > 0) {
                    LiteGraph.pointerListenerAdd(app.canvasEl, "move", onCanvasPointerMove);
                } else {
                    LiteGraph.pointerListenerRemove(app.canvasEl, "move", onCanvasPointerMove);
                }
                userTimeout = value;
            },
        });
    },
});