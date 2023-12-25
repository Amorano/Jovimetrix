/**
 * File: a_dynamic.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'

const _prefix = '➡️'

export const NODE_DYNAMIC_IN = {
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const input_count = 0;
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
            this.addInput(_prefix, '*')
            return r
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (type, input_count, connected, link_info) {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
            util.dynamic_connection(this, input_count, connected, _prefix, '*')
            if (link_info) {
                const fromNode = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                )
                const type = fromNode.outputs[link_info.origin_slot].type
                this.inputs[input_count].type = type
            }
            if (!connected) {
                this.inputs[input_count].type = '*'
                this.inputs[input_count].label = `${_prefix}_${String.fromCharCode('A'.charCodeAt(0) + input_count)}`;
                //this.inputs[input_count].label = `${_prefix}_${input_count + 1}`
            }
        }
    }
}
