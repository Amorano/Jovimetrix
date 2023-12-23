/**
 * File: core.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"

const core = {
	name: 'jovimetrix.core',
    async setup(app) {
        const origProcessKey = LGraphCanvas.prototype.processKey;
		LGraphCanvas.prototype.processKey = function(e) {
            console.info(e)
			if (e.key === 'g') {
                console.info('asd')
				for (const id in app.selected_nodes) {
					console.info(app.selected_nodes[id]);
				}
                this.graph.change();
                e.preventDefault();
				e.stopImmediatePropagation();
				return false;
			}
			return origProcessKey.apply(this, arguments);
		}
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function() {
            // if (this.title.length > 6) this.title = this.title.substring(5, 6);
            // this.size = [250, 32];
            //this.flags.collapsed = true;
            onDrawBackground?.apply(this,arguments);
            this.onDrawBackground = onDrawBackground;
        }
    }
}

app.registerExtension(core)
