/**
 * File: akashic.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from '../core/util.js'
import { JImageWidget } from '../widget/widget_jimage.js'
import { JStringWidget } from '../widget/widget_jstring.js'

const _prefix = 'jovi'
const _id = "AKASHIC (JOV) 📓"

function get_position_style(ctx, width, y, height) {
    const MARGIN = 4;
    const elRect = ctx.canvas.getBoundingClientRect();
    const transform = new DOMMatrix()
        .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
        .multiplySelf(ctx.getTransform())
        .translateSelf(MARGIN, MARGIN + y);

    return {
        transformOrigin: '0 0',
        transform: transform,
        left: `0px`,
        top: `0px`,
        position: "absolute",
        maxWidth: `${width - MARGIN * 2}px`,
        maxHeight: `${height - MARGIN * 2}px`,
        width: `auto`,
        height: `auto`,
    };
}

const ext = {
	name: 'jovimetrix.akashic',
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === _id) {
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
                            JStringWidget(app, `${_prefix}_${index}`, util.escapeHtml(txt))
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

            this.setSize(this.computeSize())
            this.onRemoved = function () {
                for (let y in this.widgets) {
                    if (this.widgets[y].canvas) {
                        this.widgets[y].canvas.remove()
                    }
                    shared.cleanupNode(this)
                    this.widgets[y].onRemoved?.()
                }
            }
            }
	    }
    }
}

app.registerExtension(ext)
