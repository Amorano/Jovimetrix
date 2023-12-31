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
        if (nodeData.name !== _id) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            let combo_current = "NONE";
            console.info(this)
            let combo = this.widgets[0];
            let old_x = this.widgets[1]?.value || 0;
            let old_y = this.widgets[2]?.value || 0;
            let old_z = this.widgets[3]?.value || 0;
            let old_w = this.widgets[4]?.value || 0;
            let old_x_bool;
            let old_x_int;
            let old_x_str;
            combo.callback = () => {
                if (combo_current != combo.value)  {
                    old_x = this.widgets[1]?.value || old_x;
                    // old_x_bool = old_x_bool || old_x;
                    // old_x_int = old_x_int || old_x;
                    // old_x_str = old_x_str || old_x;
                    old_y = this.widgets[2]?.value || old_y;
                    old_z = this.widgets[3]?.value || old_z;
                    old_w = this.widgets[4]?.value || old_w;

                    if (combo_current == 'BOOLEAN') {
                        old_x_bool = this.widgets[1]?.value || old_x_bool;
                    } else if (combo_current == 'INT') {
                        old_x_int = this.widgets[1]?.value || old_x;
                    } else if (combo_current == 'STRING') {
                        old_x_str = this.widgets[1]?.value || old_x_str;
                    }
                    console.info(old_x_bool, old_x_int, old_x_str, old_x, old_y, old_z, old_w);

                    while ((this.widgets || [])[1]) {
                        util.removeWidget(this, 1);
                    }

                    if (this.outputs && this.outputs.length > 0) {
                        this.removeOutput(0)
                    }

                    if (combo.value == 'BOOLEAN') {
                        ComfyWidgets[combo.value](this, '🇽', [combo.value, {"default": old_x_bool}], app)
                    } else if (combo.value == 'INT') {
                        ComfyWidgets[combo.value](this, '🇽', [combo.value, {"default": old_x_int, "step": 1}], app)
                    } else if (combo.value == 'FLOAT') {
                        ComfyWidgets[combo.value](this, '🇽', [combo.value, {"default": old_x, "step": 0.01}], app)
                    } else if (combo.value == 'STRING') {
                        ComfyWidgets[combo.value](this, '🇽', [combo.value, {"default": old_x_str}], app)
                    } else {
                        ComfyWidgets.FLOAT(this, '🇽', ["FLOAT", {"default": old_x, "step": 0.01}], app)
                        if (combo.value === "VEC2") {
                            ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": old_y, "step": 0.01}], app)
                        }
                        else if (combo.value === "VEC3") {
                            ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": old_y, "step": 0.01}], app)
                            ComfyWidgets.FLOAT(this, '🇿', ["FLOAT", {"default": old_z, "step": 0.01}], app)
                        }
                        else if (combo.value === "VEC4") {
                            ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": old_y, "step": 0.01}], app)
                            ComfyWidgets.FLOAT(this, '🇿', ["FLOAT", {"default": old_z, "step": 0.01}], app)
                            ComfyWidgets.FLOAT(this, '🇼', ["FLOAT", {"default": old_w, "step": 0.01}], app)
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
                    this.addOutput(map[combo.value], combo.value, { shape: LiteGraph.CIRCLE_SHAPE });
                    combo_current = combo.value;
                }
            }
            setTimeout(() => { combo.callback(); }, 15);
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (side, slot, connect, link_info, output) {
            // Logger.trace("onConnectionsChange", arguments, this);
            console.info(side, slot, connect, link_info, output)
            console.info(output.links)
            if(!link_info || side == 2)
                return;
            onConnectionsChange?.apply(this, arguments);
            this.onResize?.(this.size);
        };
	}
}

app.registerExtension(ext)
