/**
 * File: akashic.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { fitHeight, node_cleanup } from '../core/util.js'
import * as util_dom from '../core/util_dom.js'
import { JImageWidget } from '../widget/widget_jimage.js'
import { JStringWidget } from '../widget/widget_jstring.js'

const _prefix = 'jovi'
const _id = "AKASHIC (JOV) 📓"

const ext = {
	name: 'jovimetrix.node.akashic',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments)
            if (this.widgets) {
                for (let i = 0; i < this.widgets.length; i++) {
                    this.widgets[i].onRemoved?.()
                }
                this.widgets.length = 1
            }

            let index = 0
            if (message.text) {
                for (const txt of message.text) {
                    const w = this.addCustomWidget(
                        JStringWidget(app, `${_prefix}_${index}`, util_dom.escapeHtml(txt))
                    )
                    w.parent = this
                    index++
                }
            }

            if (message.b64_images) {
                for (const img of message.b64_images) {
                    const w = this.addCustomWidget(
                        JImageWidget(app, `${_prefix}_${index}`, img)
                    )
                    w.parent = this
                    index++
                }
            }

            fitHeight(this);
            this.onRemoved = function () {
                for (let y in this.widgets) {
                    if (this.widgets[y].canvas) {
                        this.widgets[y].canvas.remove()
                    }
                    node_cleanup(this)
                    this.widgets[y].onRemoved?.()
                }
            }
        }
    }
}

app.registerExtension(ext)
