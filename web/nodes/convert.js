/**
 * File: convert.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'

const _id = "CONVERT (JOV) 🧬"

const ext = {
	name: 'jovimetrix.node.convert',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;

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

            this.onConnectionsChange = function(slotType, slot, event, link_info, output) {
                if (slotType === util.TypeSlot.Input && event === util.TypeSlotEvent.Disconnect) {
                    this.inputs[slot].type = '*';
                    this.inputs[slot].name = '*';
                }

                if (link_info && slotType === util.TypeSlot.Input && event === util.TypeSlotEvent.Connect) {
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
