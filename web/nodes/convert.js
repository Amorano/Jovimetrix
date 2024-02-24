/**
 * File: convert.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
// import { TypeSlot, TypeSlotEvent } from '../util/util.js'
import { node_add_dynamic } from '../util/util.js'

const _id = "CONVERT (JOV) 🧬"
const _prefix = '🧬'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        // nodeType = node_add_dynamic(nodeType, _prefix);

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const combo = this.widgets.find(w => w.name === '❓');
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
})
