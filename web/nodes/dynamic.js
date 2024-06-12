/**
 * File: dynamic.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'

const _id = "DYNAMIC (JOV) 💥"
const _suffix = '🦄'

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            const val = this.widgets.find(w => w.name === 'VAL');
            const self = this;
            val.callback = (value) => {
                let size = value - self.outputs?.length || 1;
                for(let i=1; i<=size; i++) {
                    self.addOutput(_suffix, '*');
                }
                size = self.outputs?.length || 1;
                while (size >= value && size > 0) {
                    self.removeOutput(size);
                    size -= 1;
                }
                fitHeight(self);
            }
            setTimeout(() => { val.callback(); }, 10);
            return me;
        }
	}
})
