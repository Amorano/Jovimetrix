/**
 * File: core_cozy_fields.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { setting_make } from '../util/util_api.js'
import { colorHex2RGB } from '../util/util.js'

let g_color_style;
let g_thickness = 1;
let g_highlight;

app.registerExtension({
    name: "jovimetrix.cozy.fields",
    setup() {
        setting_make('color 🎨.style', 'Style', 'combo', 'Style to draw nodes.', "ComfyUI Default", {}, ["ComfyUI Default", "Round Highlight", "Line Highlight"]);
        setting_make('color 🎨.thickness', 'Style Thickness', 'number', 'Line thickness around widgets in Round or Line Highlight Mode.', 1, {
            min: -10,
            max: 3,
            step: 1,
        });
        setting_make('color 🎨.highlight', 'Style Highlight', 'number', 'Color Highlight if Round or Line mode enabled. Hex code entry #FFF. The default will use the node base color.');
    },
    async nodeCreated(node) {

        if (node.comfyClass.includes("(JOV)")) {
            if (node.widgets) {
                for (const widget of node.widgets) {
                    if (widget.name == "control_after_generate") {
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
})