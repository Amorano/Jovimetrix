/**
 * File: glsl.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js";
import * as util from '../core/util.js'

const _id = "GLSL (JOV) 🍩"

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const widget_fragment = this.widgets[5];
            const shaders = await util.api_get("/jovimetrix/glsl");
            const widget_preset = this.addWidget("combo", "PRESET", "", (e) => {
                widget_fragment.value = shaders[e];
            },
            {
                values: () => {
                    const sortedKeys = Object.keys(shaders).sort();
                    return sortedKeys;
                }
            },
            {
                serialize: false
            });

            this.addWidget("button", "SAVE PROGRAM", "", async (e) => {
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
                        name: value,
                        data: widget_fragment.value
                    };
                    util.api_post("/jovimetrix/glsl", data);
                }
            });
            return me;
        }
    }
}



app.registerExtension(ext)
