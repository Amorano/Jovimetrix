/**
 * File: value.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { ComfyWidgets } from "/scripts/widgets.js"
import { widget_remove } from '../util/util_widget.js'
import { fitHeight } from '../util/util.js'

const _id = "VALUE (JOV) #️⃣"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;
            let combo_current = "";
            const widget_x = this.widgets.find(w => w.name === '🇽');
            const widget_y = this.widgets.find(w => w.name === '🇾');
            const widget_z = this.widgets.find(w => w.name === '🇿');
            const widget_w = this.widgets.find(w => w.name === '🇼');
            const combo = this.widgets.find(w => w.name === '❓');
            let old_x = widget_x?.value || 0;
            let old_y = widget_y?.value || 0;
            let old_z = widget_z?.value || 0;
            let old_w = widget_w?.value || 0;
            let old_x_bool;
            let old_x_str;
            combo.callback = () => {
                if (combo_current != combo.value)  {
                    old_x = widget_x?.value || old_x;
                    old_y = widget_y?.value || old_y;
                    old_z = widget_z?.value || old_z;
                    old_w = widget_w?.value || old_w;
                    if (combo_current == 'BOOLEAN') {
                        old_x_bool = widget_x?.value || old_x_bool;
                    } else if (combo_current == 'STRING') {
                        old_x_str = widget_x?.value || old_x_str;
                    }
                    // remember the connections and attempt to re-connect
                    /*
                    let old_connect = [];
                    if (this.outputs && this.outputs.length > 0) {
                        const old = this.outputs[0].links ? this.outputs[0].links.map(x => x) : [];
                        for (const id of old) {
                            var link = this.graph.links[id];
                            if (!link) {
                                continue;
                            }
                            var node = this.graph.getNodeById(link.target_id);
                            if (node) {
                                old_connect.push({
                                    node: node,
                                    slot: link.target_slot,
                                })
                            }
                        }
                        this.removeOutput(0);
                    }*/

                    while ((this.widgets || [])[1]) {
                        widget_remove(this, 1);
                    }
                    const x = this.inputs.find(w => w.name === '🇽');
                    const y = this.inputs.find(w => w.name === '🇾');
                    const z = this.inputs.find(w => w.name === '🇿');
                    const w = this.inputs.find(w => w.name === '🇼');
                    if (combo.value == 'BOOLEAN' && x == undefined) {
                        ComfyWidgets[combo.value](this, '🇽', [combo.value, {"default": old_x_bool}], app)
                    } else if (combo.value == 'INT' && x == undefined) {
                        ComfyWidgets[combo.value](this, '🇽', [combo.value, {"default": old_x, "step": 1}], app)
                    } else if (combo.value == 'FLOAT' && x == undefined) {
                        ComfyWidgets[combo.value](this, '🇽', [combo.value, {"default": old_x, "step": 0.01}], app)
                    } else if (combo.value == 'STRING' && x == undefined) {
                        ComfyWidgets[combo.value](this, '🇽', [combo.value, {"default": old_x_str, "multiline": true, "dynamicPrompts": false}], app)
                    } else {
                        if (combo.value === "VEC2") {
                            if (x == undefined) {
                                ComfyWidgets.FLOAT(this, '🇽', ["FLOAT", {"default": old_x, "step": 0.01}], app);
                            }
                            if (y == undefined) {
                                ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": old_y, "step": 0.01}], app)
                            }
                        } else if (combo.value === "VEC2INT") {
                            if (x == undefined) {
                                ComfyWidgets.INT(this, '🇽', ["INT", {"default": old_y, "step": 1}], app)
                            }
                            if (y == undefined) {
                                ComfyWidgets.INT(this, '🇾', ["INT", {"default": old_y, "step": 1}], app)
                            }
                        } else if (combo.value === "VEC3") {
                            if (x == undefined) {
                                ComfyWidgets.FLOAT(this, '🇽', ["FLOAT", {"default": old_x, "step": 0.01}], app);
                            }
                            if (y == undefined) {
                                ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": old_y, "step": 0.01}], app)
                            }
                            if (z == undefined) {
                                ComfyWidgets.FLOAT(this, '🇿', ["FLOAT", {"default": old_z, "step": 0.01}], app)
                            }
                        } else if (combo.value === "VEC3INT") {
                            if (x == undefined) {
                                ComfyWidgets.INT(this, '🇽', ["INT", {"default": old_y, "step": 1}], app)
                            }
                            if (y == undefined) {
                                ComfyWidgets.INT(this, '🇾', ["INT", {"default": old_y, "step": 1}], app)
                            }
                            if (z == undefined) {
                                ComfyWidgets.INT(this, '🇿', ["INT", {"default": old_z, "step": 1}], app)
                            }
                        } else if (combo.value === "VEC4") {
                            if (x == undefined) {
                                ComfyWidgets.FLOAT(this, '🇽', ["FLOAT", {"default": old_x, "step": 0.01}], app);
                            }
                            if (y == undefined) {
                                ComfyWidgets.FLOAT(this, '🇾', ["FLOAT", {"default": old_y, "step": 0.01}], app)
                            }
                            if (z == undefined) {
                                ComfyWidgets.FLOAT(this, '🇿', ["FLOAT", {"default": old_z, "step": 0.01}], app)
                            }
                            if (w == undefined) {
                                ComfyWidgets.FLOAT(this, '🇼', ["FLOAT", {"default": old_w, "step": 0.01}], app)
                            }
                        } else if (combo.value === "VEC4INT") {
                            if (x == undefined) {
                                ComfyWidgets.INT(this, '🇽', ["INT", {"default": old_y, "step": 1}], app)
                            }
                            if (y == undefined) {
                                ComfyWidgets.INT(this, '🇾', ["INT", {"default": old_y, "step": 1}], app)
                            }
                            if (z == undefined) {
                                ComfyWidgets.INT(this, '🇿', ["INT", {"default": old_z, "step": 1}], app)
                            }
                            if (w == undefined) {
                                ComfyWidgets.INT(this, '🇼', ["INT", {"default": old_w, "step": 1}], app)
                            }
                        }
                    }

                    const my_map = {
                        STRING: "📝",
                        BOOLEAN: "🇴",
                        INT: "🔟",
                        FLOAT: "🛟",
                        VEC2: "🇽🇾",
                        VEC2INT: "🇽🇾",
                        VEC3: "🇽🇾\u200c🇿",
                        VEC3INT: "🇽🇾\u200c🇿",
                        VEC4: "🇽🇾\u200c🇿\u200c🇼",
                        VEC4INT: "🇽🇾\u200c🇿\u200c🇼",
                    }
                    // console.debug(this.outputs[0])
                    this.outputs[0].name = my_map[combo.value];
                    combo_current = combo.value;
                }
                fitHeight(self);
            }
            setTimeout(() => { combo.callback(); }, 15);
            return me;
        }
	}
})
