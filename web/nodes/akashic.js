/**
 * File: akashic.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { fitHeight, node_cleanup } from '../util/util.js'
import { escapeHtml } from '../util/util_dom.js'
import { JImageWidget } from '../widget/widget_jimage.js'
import { JStringWidget } from '../widget/widget_jstring.js'

const _prefix = 'jovi'
const _id = "AKASHIC (JOV) 📓"

app.registerExtension({
	name: 'jovimetrix.node.' + _id,
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
            console.debug("unknown message", message)
            if (message.text && message.txt != "") {
                for (const txt of message.text) {
                    const w = this.addCustomWidget(
                        JStringWidget(app, `${_prefix}_${index}`, escapeHtml(txt))
                    )
                    w.parent = this
                    index++
                }
            }
            else if (message.b64_images) {
                for (const img of message.b64_images) {
                    const w = this.addCustomWidget(
                        JImageWidget(app, `${_prefix}_${index}`, img)
                    )
                    w.parent = this
                    index++
                }
            } else {
                console.debug("unknown message", message)
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
})
