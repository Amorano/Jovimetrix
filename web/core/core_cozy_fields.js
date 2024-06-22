/**
 * File: cozy_fields.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import * as util_config from '../util/util_config.js'
import { hex2rgb } from '../util/util_color.js'
import { $el } from "../../../scripts/ui.js"
import { node_isOverInput } from '../util/util.js'

const style = `
.tooltips {
    object-fit: absolute;
    width: var(--comfy-img-preview-width);
    height: var(--comfy-img-preview-height);
}
`;

let g_color_style;
let g_thickness = 1;
let g_highlight;

app.registerExtension({
    name: "jovimetrix.cozy.fields",
    init() {
        let id = "user.default.color.style"
        app.ui.settings.addSetting({
            id: `jov.${id}`,
            name: "🇯 🎨 Node Color Style",
            type: "combo",
            options: ["ComfyUI Default", "Round Highlight", "Line Highlight"],
            tooltip: "Style to draw nodes.",
            defaultValue: "ComfyUI Default",
            onChange: function(val) {
                util_config.setting_store(id, val);
                g_color_style = val;
            },
        });
        id = "user.default.color.thickness"
        app.ui.settings.addSetting({
            id: `jov.${id}`,
            name: "🇯 🎨 Node Color Style Thickness",
            type: "number",
            attrs: {
                min: -10,
                max: 3,
                step: 1,
            },
            tooltip: "Line thickness around widgets in Round or Line Highlight Mode.",
            defaultValue: 1,
            onChange: function(val) {
                util_config.setting_store(id, val);
                g_thickness = val;
            },
        });
        id = "user.default.color.highlight"
        app.ui.settings.addSetting({
            id: `jov.${id}`,
            name: "🇯 🎨 Node Color Style Highlight",
            type: "string",
            tooltip: "Line thickness around widgets in Round or Line Highlight Mode.",
            defaultValue: "",
            onChange: function(val) {
                util_config.setting_store(id, val);
                g_highlight = val;
            },
        });
    },
    async setup() {
        $el(
            'div',
            {
                id: "jov-tooltip",
                parent: document.body,
            },
            [
                $el("table", [
                    $el(
                        "caption",
                        { textContent: "Settings" },
                    ),
                    $el("button", {
                        type: "button",
                        textContent: "Close",
                        style: {
                            cursor: "pointer",
                        },
                    }),
                ]),
            ]
        );
    },
    async nodeCreated(node) {
        const onDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function (ctx, area) {
            // console.info(node)
            const me = onDrawForeground?.apply(this, arguments);
            if (this.widgets) {
                ctx.save();
                // @TODO: Switch to USER choice here....
                if (g_color_style) {
                    let color = ctx.strokeStyle;
                    if (g_highlight != "") {
                        try {
                            color = hex2rgb(g_highlight);
                            color = g_highlight
                        } catch { }
                    }
                    if (g_color_style == "Round Highlight") {
                        const thick = Math.max(1, Math.min(3, g_thickness));

                        for (const w of this.widgets) {
                            if (w?.hidden || w.type.startsWith('converted-') || ["customtext"].includes(w.type)) {
                                continue;
                            }
                            ctx.beginPath();
                            ctx.fillStyle = color;
                            ctx.roundRect(15-thick, w.last_y-thick, node.size[0]-30+2*thick, LiteGraph.NODE_WIDGET_HEIGHT+thick * 2, 12);

                            ctx.fill();
                        }
                    } else if (g_color_style == "Line Highlight") {
                        for (const w of this.widgets) {
                            if (w?.hidden || w.type.startsWith('converted-') || ["customtext"].includes(w.type)) {
                                continue;
                            }
                            ctx.beginPath();
                            ctx.fillStyle = color;
                            ctx.rect(0, w.last_y-g_thickness, node.size[0], LiteGraph.NODE_WIDGET_HEIGHT+g_thickness * 2);
                            ctx.fill();
                        }
                    }
                }
                ctx.restore();
            }
            return me;
        }

        const widget_tooltip = (node.widgets || [])
            .find(widget => widget.type === 'JTOOLTIP');
        if (widget_tooltip) {
            let hoverTimeout = null;
            let last_control = null;
            const tips = widget_tooltip.options.default || {};
            const hoverThreshold = 220;
            const onMouseMove = node.onMouseMove;
            node.onMouseMove = function (e, pos, graph) {
                const me = onMouseMove?.apply(this, arguments);
                if (hoverTimeout) {
                    clearTimeout(hoverTimeout);
                    hoverTimeout = null;
                }

                if (!node.flags.collapsed) {
                    const slot = node.getSlotInPosition(pos[0] + node.pos[0], pos[1] + node.pos[1]);
                    hoverTimeout = setTimeout(() => {
                        if (slot) {
                            if (last_control != slot) {
                                let tip;
                                if (slot.input) {
                                    tip = tips?.[slot.input.name];
                                } else if (slot.output) {
                                    tip = tips?.[slot.output.name];
                                } else if (slot.widgets) {
                                    tip = tips?.[slot.widgets.name];
                                }
                                if (tip) {
                                    console.info(tip)
                                    //tooltip.style.left = `${e.clientX}px`;
                                    //tooltip.style.top = `${e.clientY + 20}px`;
                                    //tooltip.style.display = 'block';
                                    //tooltip.innerHTML = 'Hovered over target for more than 1 second';
                                }
                            }
                            last_control = slot;
                        }
                    }, hoverThreshold);
                }
                return me;
            }
        }
    },
})
