/**
 * File: colorize.js
 * Project: Jovimetrix
 * Author: Alexander G. Morano
 *
 * Copyright (c) 2023 Alexander G. Morano
 *
 */

import { api } from "../../../../scripts/api.js";
import { app } from "../../../../scripts/app.js";

var response = await api.fetchApi("/jovimetrix/config/raw", { cache: "no-store" });
const JOV_CONFIG = await response.json();

app.registerExtension({
    name: "jovimetrix.colorize",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        //console.log(JOV_CONFIG)
        if (JOV_CONFIG.color === undefined || JOV_CONFIG.color.length == 0) {
            return;
        }

        let found = JOV_CONFIG.color[nodeData.type || nodeData.name];
        if (found === undefined && nodeData.category) {
            console.log(nodeData.type || nodeData.name)
            const categorySegments = nodeData.category.split('/');
            let k = categorySegments.join('/');

            while (k) {
                found = JOV_CONFIG.color[k];
                if (found) {
                    break;
                }
                const lastSlashIndex = k.lastIndexOf('/');
                k = lastSlashIndex !== -1 ? k.substring(0, lastSlashIndex) : '';
            }
        }

        if (found) {
            // console.log(nodeData);
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                if (nodeData.color === undefined) {
                    this['color'] = (found.title || "#7F7F7FFF")
                }

                if (nodeData.bgcolor === undefined) {
                    this['bgcolor'] = (found.body || "#7F7F7FFF")
                }

                /*
                // default, box, round, card
                if (nodeData.shape === undefined || nodeData.shape == false) {
                    this['_shape'] = nodeData._shape ? nodeData._shape : found.shape ?
                    found.shape in ['default', 'box', 'round', 'card'] : 'round';
                }*/
                // console.debug("jovi-colorized", this.title, this.color, this.bgcolor, this._shape);
                // result.serialize_widgets = true
                return result;
            }
        }
    },
})