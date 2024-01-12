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

            const shaders = await util.api_get("/jovimetrix/glsl");
            const widget_fragment = this.widgets[5];
            const widget_preset = this.widgets[8];
            widget_preset.callback = (e) => {
                widget_fragment.value = shaders[e];
            }
            widget_preset.options.values = () => {
                return Object.keys(shaders).sort();
            }
            widget_preset.serialize = false;

            async function saveProgram() {
                try {
                    const value = await util.showModal(
                        `<div class="jov-modal-content">
                            <h2>SAVE GLSL PROGRAM</h2>
                            <input type="text" id="filename" placeholder="filename">
                            <button id="jov-submit-continue">SAVE</button>
                            <button id="jov-submit-cancel">CANCEL</button>
                        </div>`,
                        (button) => {
                            if (button === "jov-submit-continue") {
                                return document.getElementById("filename").value;
                            } else if (button === "jov-submit-cancel") {
                                return "";
                            }
                        }
                    );

                    if (value) {
                        const data = {
                            name: value,
                            data: widget_fragment.value
                        };
                        util.api_post("/jovimetrix/glsl", data);
                    }
                } catch (error) {
                    console.error(error);
                }
            }

            this.addWidget("button", "SAVE PROGRAM", null, (e) => {
                saveProgram();
            });
            return me;
        }

        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const widget_time = this.widgets[0];
            const widget_fps = this.widgets[1];
            const widget_batch = this.widgets[2];
            const offset = widget_fps.value / 1000 * widget_batch.value;
            widget_time.value += offset;
        }
    }
}

app.registerExtension(ext)
