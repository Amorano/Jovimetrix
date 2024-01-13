/**
 * File: queue.js
 * Project: Jovimetrix
 *
 */

import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import * as util from '../core/util.js'
import * as fun from '../core/fun.js'

const _id = "QUEUE (JOV) 🗃"

const ext = {
	name: _id,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _id) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            const self = this;
            const widget_queue = this.widgets[0];
            async function python_queue_ping(event) {
                if (event.detail.id != self.id) {
                    return;
                }

                // fun.bewm(self.pos[0], self.pos[1]);
            }

            async function python_queue_done(event) {
                if (event.detail.id != self.id) {
                    return;
                }
                let centerX = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
                let centerY = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;


                fun.bewm(centerX / 2, centerY / 3);
                await util.flashBackgroundColor(widget_queue.inputEl, 250, 5,  "#449262AA");
                await util.flashBackgroundColor(widget_queue.inputEl, 450, 4,  "#667252BB");
                await util.flashBackgroundColor(widget_queue.inputEl, 850, 3,  "#995242CC");
                await util.flashBackgroundColor(widget_queue.inputEl, 1650, 2, "#BB3232DD");
                await util.flashBackgroundColor(widget_queue.inputEl, 3500, 1, "#FF2222EE");
            }
            api.addEventListener("jovi-queue-ping", python_queue_ping);
            api.addEventListener("jovi-queue-done", python_queue_done);
            return me;
        }
    }
}
app.registerExtension(ext)
