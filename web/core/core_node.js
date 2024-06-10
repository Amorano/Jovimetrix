/**
 * Re-order the graph into a linear line to analyze
 */

import { app } from "../../../scripts/app.js"
import { CONVERTED_TYPE } from '../util/util_widget.js'
import { CONFIG_USER } from '../util/util_config.js'
import { node_cleanup } from '../util/util.js'

app.registerExtension({
    name: "jovimetrix.node_ext",
    async getCustomWidgets(app) {

    },
    init() {

	},
    handleKeydown(e) {
        if ((e.altKey && e.kKey) || (e.ctrlKey && e.shiftKey)) {
            something
        };
    },
	beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.category?.startsWith("JOVIMETRIX")) {
            return;
        }
	}
})
