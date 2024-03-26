/**
 * File: workflow.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight } from '../util/util.js'
import { widget_hide, widget_show } from '../util/util_widget.js'

const _id = "WORKFLOW (JOV) ➿"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = async function () {
            const me = onExecuted?.apply(this);
            let filename = "workflow.json";

            if (!filename.toLowerCase().endsWith(".json")) {
                filename += ".json";
            }

            app.graphToPrompt().then(p=>{
                // convert the data to a JSON string
                const json = JSON.stringify(p.workflow, null, 2);
                const blob = new Blob([json], {type: "application/json"});
                const fileWriter = new FileWriter();
                fileWriter.write(blob, filename);
            });
            return me;
        }
    }
})
