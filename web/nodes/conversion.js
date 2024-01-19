/**
 * File: conversion.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from '../core/util.js'

const _id = "CONVERT (JOV) 🧬"
const _cat = "JOVIMETRIX 🔺🟩🔵/CALC"

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
	name: 'jovimetrix.node.convert',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            let combo_current = "NONE";
            // console.debug("jovimetrix.node.convert.onNodeCreated", this)
            let combo = this.widgets[0]
            combo.callback = () => {
                if (combo_current != combo.value)  {
                    if (this.outputs && this.outputs.length > 0) {
                        this.removeOutput(0)
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
                this.onResize?.(this.size);
            }
            setTimeout(() => { combo.callback(); }, 15);

            this.onConnectionsChange = function(slotType, slot, isChangeConnect, link_info, output) {
                if (slotType === util.TypeSlot.Input && isChangeConnect === util.TypeSlotEvent.Disconnect) {
                    this.inputs[slot].type = '*';
                    this.inputs[slot].name = '*';
                }

                if (link_info && slotType === util.TypeSlot.Input && isChangeConnect === util.TypeSlotEvent.Connect) {
                    const fromNode = this.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
                    const type = fromNode.outputs[link_info.origin_slot].type;
                    this.inputs[0].type = type;
                    this.inputs[0].name = type;
                }
            }

            return me;
        }
    }
}

app.registerExtension(ext)
