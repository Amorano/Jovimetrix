/**
 * File: lerp
 * .js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { widgetABHook2, widgetOutputHookType } from '../util/util_jov.js'

const _id = "LERP (JOV) 🔰"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            const alpha = this.widgets.find(w => w.name === '🛟');
            const AA = this.widgets.find(w => w.name === '🅰️🅰️');
            const BB = this.widgets.find(w => w.name === '🅱️🅱️');
            const combo = this.widgets.find(w => w.name === '❓');
            widgetABHook2(this, '❓', alpha, true);
            widgetABHook2(this, '❓', AA);
            widgetABHook2(this, '❓', BB);
            widgetOutputHookType(this, '❓');
            setTimeout(() => { combo.callback(); }, 5);
            return me;
        }
        return nodeType;
	}
})
