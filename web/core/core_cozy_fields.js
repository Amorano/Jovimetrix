/**
 * File: cozy_fields.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import * as util_config from '../util/util_config.js'
import { hex2rgb } from '../util/util_color.js'

let g_color_style;
let g_thickness = 1;
let g_highlight;

app.registerExtension({
    name: "jovimetrix.cozy.fields",
    setup() {
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
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated.apply(this, arguments);
            if (this.widgets) {
                for (const widget of this.widgets) {
                    if (widget.name === "control_after_generate") {
                        widget.value = "fixed";
                    }
                }
            }
            return me;
        }
    },
    async nodeCreated(node) {
        const onDrawForeground = node.onDrawForeground;
        node.onDrawForeground = async function (ctx, area) {
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
    },
})
