/**
 * File: core_cozy_fields.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import * as util_config from '../util/util_config.js'
import { colorHex2RGB } from '../util/util.js'

let g_color_style;
let g_thickness = 1;
let g_highlight;

app.registerExtension({
    name: "jovimetrix.cozy.fields",
    setup() {
        let id = "color 🎨.style"
        app.ui.settings.addSetting({
            id: `JOVIMETRIX 🔺🟩🔵.${id}`,
            name: "Style",
            type: "combo",
            options: ["ComfyUI Default", "Round Highlight", "Line Highlight"],
            tooltip: "Style to draw nodes.",
            defaultValue: "ComfyUI Default",
            onChange: function(val) {
                util_config.setting_store(id, val);
                g_color_style = val;
            },
        });


        id = "color 🎨.thickness"
        app.ui.settings.addSetting({
            id: `JOVIMETRIX 🔺🟩🔵.${id}`,
            name: "Style Thickness",
            type: "number",
            attrs: {
                min: -10,
                max: 3,
                step: 1,
            },
            tooltip: "Line thickness around widgets in Round or Line Highlight Mode.",
            defaultValue: 1,
            onChange: function(val) {
                setting_store(id, val);
                g_thickness = val;
            },
        });


        id = "color 🎨.highlight"
        app.ui.settings.addSetting({
            id: `JOVIMETRIX 🔺🟩🔵.${id}`,
            name: "Style Highlight",
            type: "string",
            tooltip: "Color Highlight if Round or Line mode enabled. Hex code entry #FFF. The default will use the node base color.",
            defaultValue: "",
            onChange: function(val) {
                setting_store(id, val);
                g_highlight = val;
            },
        });
    },
    async nodeCreated(node) {

        if (node.comfyClass.includes("(JOV)")) {
            if (node.widgets) {
                for (const widget of node.widgets) {
                    if (widget.name === "control_after_generate") {
                        widget.value = "fixed";
                    }
                }
            }
        }

        const onDrawForeground = node.onDrawForeground;
        node.onDrawForeground = async function (ctx) {
            const me = onDrawForeground?.apply(this, arguments);
            if (this.widgets) {
                ctx.save();
                // @TODO: Switch to USER choice here....
                if (g_color_style) {
                    let color = ctx.strokeStyle;
                    if (g_highlight != "") {
                        try {
                            color = colorHex2RGB(g_highlight);
                            color = g_highlight
                        } catch {
                            // Intentionally left blank
                        }
                    }
                    if (g_color_style == "Round Highlight") {
                        const thick = Math.max(1, Math.min(3, g_thickness));

                        for (const w of this.widgets) {
                            if (!w || w?.hidden || !w?.type) {
                                continue
                            }

                            if (w.type.startsWith('converted-') || ["customtext"].includes(w.type)) {
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
    }
    */
})