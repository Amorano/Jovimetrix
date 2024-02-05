/**
 * File: select.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'
import { ComfyWidgets } from "/scripts/widgets.js"

const _id = "SELECT (JOV) 🤏🏽"
const _prefix = '❔'

const ext = {
	name: 'jovimetrix.node.select',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this)
            const self = this;
            this.input_count = 1;
            const widget_reset = this.widgets[1];
            widget_reset.callback = async (e) => {
                widget_reset.value = false;
                util.api_cmd_jovian(self.id, "reset");
            }
            this.addInput(`${_prefix}_1`, '*');
            return me
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
            const me = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
            if (slotType === util.TypeSlot.Input) {
                util.dynamic_connection(this, slot, event, `${_prefix}_`, '*')
                if (event === util.TypeSlotEvent.Connect && link_info) {
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    )
                    const type = fromNode.outputs[link_info.origin_slot].type
                    this.inputs[slot].type = type
                } else if (event === util.TypeSlotEvent.Disconnect) {
                    this.inputs[slot].type = '*'
                    this.inputs[slot].label = `${_prefix}_${slot + 1}`
                }
            }
            return me;
        }
	}
}

app.registerExtension(ext)
