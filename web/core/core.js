/**
 * File: core.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from './util.js'

const core = {
	name: 'jovimetrix.core',
    async beforeRegisterNodeDef_not(nodeType, nodeData, app) {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            let bgcolor;
            let startTime;
            const fadeDuration = 1500;

            const updateColor = () => {
                const elapsed = Date.now() - startTime;
                const lerp = Math.min(1, elapsed / fadeDuration);
                this.bgcolor = util.fade_lerp_color("#FFFFFF", bgcolor, lerp);
                this.graph.setDirtyCanvas(true, false);
                if (lerp >= 1 || elapsed > fadeDuration) {
                    // console.info(this.bgcolor)
                    clearInterval(this._fadeIntervalId);
                    this.bgcolor = bgcolor;
                    return;
                }
            };

            const onExecutionStart = nodeType.prototype.onExecutionStart;
            nodeType.prototype.onExecutionStart = function (message) {
                onExecutionStart?.apply(this, arguments);
                startTime = Date.now();
                bgcolor = this.bgcolor || "#13171D";
                //this.bgcolor = "#FFFFFF";
                this._fadeIntervalId = setInterval(updateColor, 20);
            };
            return me;
        };
	}
}

app.registerExtension(core)
