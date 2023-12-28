/**
 * File: value.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from '../core/util.js'
import { ComfyWidgets } from "/scripts/widgets.js"

const _id = "VALUE (JOV) #️⃣"

function get_position_style(ctx, widget_width, y, node_height) {
    const MARGIN = 4;
    const elRect = ctx.canvas.getBoundingClientRect();
    const transform = new DOMMatrix()
        .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
        .multiplySelf(ctx.getTransform())
        .translateSelf(MARGIN, MARGIN + y);

    return {
        transformOrigin: '0 0',
        transform: transform,
        left: `0px`,
        top: `0px`,
        position: "absolute",
        maxWidth: `${widget_width - MARGIN * 2}px`,
        maxHeight: `${node_height - MARGIN * 2}px`,
        width: `${ctx.canvas.width}px`,  // Set canvas width
        height: `${ctx.canvas.height}px`,  // Set canvas height
    };
}

const ext = {
	name: 'jovimetrix.node.value',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === _id) {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const me = onNodeCreated?.apply(this)
                // add output based on the type selected in Lexicon.TYPE => ❓
                // util.removeWidgets(this);
                const dropdown = [["STRING", "BOOLEAN", "INT", "FLOAT", "VEC2", "VEC3", "VEC4", ], {"default": "BOOLEAN"}]
                let combo_current = "NONE";
                const combo = app.widgets.COMBO(this, "❓", dropdown, app)
                combo.widget.callback = () => {
                    if (combo_current != combo.widget.value)  {
                        while ((this.widgets || [])[1]) {
                            util.removeWidget(this, 1);
                        }
                        console.info(this)
                        if (this.outputs && this.outputs.length > 0) {
                            this.removeOutput(0)
                        }
                    }
                    if (['BOOLEAN', 'FLOAT', 'INT', 'STRING'].includes(combo.widget.value)) {
                        ComfyWidgets[combo.widget.value](this, '🇽', [combo.widget.value, {"default": 0, "step": 0.01}], app)
                    } else {
                        ComfyWidgets.FLOAT(this, '🇽', ["FLOAT", {"default": 0, "step": 0.01}], app)
                        if (combo.widget.value === "VEC2") {
                            ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": 0, "step": 0.01}], app)
                        }
                        else if (combo.widget.value === "VEC3") {
                            ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": 0, "step": 0.01}], app)
                            ComfyWidgets.FLOAT(this, '🇿', ["FLOAT", {"default": 0, "step": 0.01}], app)
                        }
                        else if (combo.widget.value === "VEC4") {
                            ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": 0, "step": 0.01}], app)
                            ComfyWidgets.FLOAT(this, '🇿', ["FLOAT", {"default": 0, "step": 0.01}], app)
                            ComfyWidgets.FLOAT(this, '🇼', ["FLOAT", {"default": 0, "step": 0.01}], app)
                        }
                    }
                    const map = {
                        STRING: "📝",
                        BOOLEAN: "🇴",
                        INT: "🔟",
                        FLOAT: "🛟",
                        VEC2: "🇽🇾",
                        VEC3: "🇽🇾\u200c🇿",
                        VEC4: "🇽🇾\u200c🇿\u200c🇼",
                    }
                    this.addOutput(map[combo.widget.value], combo.widget.value, { shape: LiteGraph.CIRCLE_SHAPE });
                    combo_current = combo.widget.value;
                };
                setTimeout(() => { combo.widget.callback(); }, 15);
                return me;
            };
        }
	}
}

app.registerExtension(ext)
