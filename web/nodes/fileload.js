/**
 * File: fileselect.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from '../core/util.js'

const _id = "FILE SELECT (JOV) 📑"

const ext = {
	name: 'jovimetrix.node.fileselect',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments)
            const self = this;
            this.addWidget("button", "LOAD DIRECTORY", "", async (e) => {
                const value = await (async () => {
                    return util.showModal(`
                        <div class="jov-modal-content">
                            <h2>CANCEL OR CONTINUE?</h2>
                            <input type="text" id="inputValue" placeholder="">
                            <button id="jov-submit">Submit</button>
                        </div>`, () => {
                            return document.getElementById("inputValue").value;
                        }
                    );
                })();

                if (value) {
                    const data = {
                        id: self.id,
                        data: widget_fragment.value
                    };
                    util.api_post("/jovimetrix/fileselect", data);
                }
            });
        }
    }
}

app.registerExtension(ext)
