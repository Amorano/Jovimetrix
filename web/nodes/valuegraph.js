/**
 * File: valuegraph.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'

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
        nodeType.prototype.onConnectionsChange = function (type, input_count, connected, link_info) {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
            util.dynamic_connection(this, input_count, connected, `${_prefix}_`, '*')
            if (link_info) {
                const fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                )
                const type = fromNode.outputs[link_info.origin_slot].type
                this.inputs[input_count].type = type
            }
            if (!connected) {

                this.inputs[input_count].type = '*'
                this.inputs[input_count].label = `${_prefix}_${input_count + 1}`
            }
        }
	}
}

app.registerExtension(ext)
