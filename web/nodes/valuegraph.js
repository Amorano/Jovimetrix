/**
 * File: valuegraph.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { TypeSlot, TypeSlotEvent, dynamic_connection } from '../core/util.js'

const _id = "VALUE GRAPH (JOV) 📈"
const _prefix = '❔'

const ext = {
	name: 'jovimetrix.node.valuegraph',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
            this.addInput(`${_prefix}_1`, '*')
            return r
        }


        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
            const me = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
            if (slotType === TypeSlot.Input) {
                dynamic_connection(this, slot, event, `${_prefix}_`, '*')
                if (event === TypeSlotEvent.Connect && link_info) {
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    )
                    const type = fromNode.outputs[link_info.origin_slot].type
                    this.inputs[slot].type = type
                } else if (event === TypeSlotEvent.Disconnect) {
                    this.inputs[slot].type = '*'
                    this.inputs[slot].label = `${_prefix}_${slot + 1}`
                }
            }
            return me;
        }
	}
}

app.registerExtension(ext)
