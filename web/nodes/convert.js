/**
 * File: convert.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
// import { TypeSlot, TypeSlotEvent } from '../core/util.js'
import { node_add_dynamic } from '../core/util.js'

const _id = "CONVERT (JOV) 🧬"
const _prefix = '🧬'

const ext = {
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        // nodeType = node_add_dynamic(nodeType, _prefix);

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            // let combo_current = "NONE";
            // console.debug("jovimetrix.node.convert.onNodeCreated", this)
            let combo = this.widgets[0]
            combo.callback = () => {
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
                this.onResize?.(this.size);
            }
            setTimeout(() => { combo.callback(); }, 15);
            return me;
        }
    }
}

app.registerExtension(ext)
