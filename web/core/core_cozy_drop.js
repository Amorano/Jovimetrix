/**
 * File: cozy_drops.js
 * Project: Jovimetrix

origin_id: 705
​origin_slot: 1
​target_id: 703
​target_slot: 0
​type: "CLIP"

 */

import { app } from "../../../scripts/app.js"
import { $el } from "../../../scripts/ui.js"
import { getHoveredWidget } from '../util/util_widget.js'

app.registerExtension({
    name: "jovimetrix.help.drop",
    setup() {

        const onCanvasPointerMove = function () {
            const link = this.over_link_center;
            if (link) {
                console.info(this.node_over)
                console.info(this.node_over)
                //console.info(this)
            }

        }.bind(app.canvas);
        LiteGraph.pointerListenerAdd(app.canvasEl, "move", onCanvasPointerMove);
        // LiteGraph.pointerListenerRemove(app.canvasEl, "move", onCanvasPointerMove);

        LiteGraph.addEventListener("dragover", this._doNothing, false);
        //LiteGraph.addEventListener("dragend", this._doNothing, false);
        //LiteGraph.addEventListener("drop", this._ondrop_callback, false);
        //LiteGraph.addEventListener("dragenter", this._doReturnTrue, false);
    }
});