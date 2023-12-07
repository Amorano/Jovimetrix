/**
 * File: colorize.js
 * Project: Jovimetrix
 * Author: Alexander G. Morano
 *
 * Copyright (c) 2023 Alexander G. Morano
 *
 */

import { app } from '../../scripts/app.js'

/**
 * Fetches JSON data from the specified file path.
 *
 * @param {string} jsonFilePath - The path to the JSON file.
 * @returns {Promise<?Object>} - A Promise that resolves to the parsed JSON data, or null in case of an error.
 */
const fetchJsonData = async (jsonFilePath) => {
    try {
        const response = await fetch(jsonFilePath);
        if (!response.ok) {
            throw new Error(`${response.statusText} ${jsonFilePath}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error loading JSON file:', error);
        return {};
    }
};

// get mine first, then update...
let user_sidecar = await fetchJsonData('./extensions/jovimetrix/user_sidecar.json');
let jovi_sidecar = await fetchJsonData('./extensions/jovimetrix/jovi_sidecar.json');

const node_data = Object.assign({}, jovi_sidecar, user_sidecar);

/**
 * My baby!
 *
 * @returns {import("./types/comfy").ComfyExtension}
 */
app.registerExtension({
    name: "jovimetrix.colorize",

    /**
    * @param {import("./types/comfy").NodeType} nodeType
    * @param {import("./types/comfy").NodeDef} nodeData
    * @param {import("./types/comfy").App} app
    */
    async beforeRegisterNodeDef(nodeType, nodeData) {

        if (node_data === undefined || node_data.length == 0) {
            return;
        }

        let found = node_data[nodeData.type || nodeData.name];
        if (found === undefined && nodeData.category) {
            console.log(nodeData.type || nodeData.name)
            const categorySegments = nodeData.category.split('/');
            let k = categorySegments.join('/');

            while (k) {
                found = node_data[k];
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
                    this['color'] = (found.title || [0, 0, 0])
                        .map(value => Math.max(0, Math.min(255, value)))
                        .map(value => value.toString(16).padStart(2, '0'));
                    this['color'] = `#${this['color'].join('')}`
                }

                if (nodeData.bgcolor === undefined) {
                    this['bgcolor'] = (found.body || [0, 0, 0])
                        .map(value => Math.max(0, Math.min(255, value)))
                        .map(value => value.toString(16).padStart(2, '0'));
                    this['bgcolor'] = `#${this['bgcolor'].join('')}`
                }

                // default, box, round, card
                if (nodeData.shape === undefined || nodeData.shape == false) {
                    this['_shape'] = nodeData._shape ? nodeData._shape : found.shape ?
                    found.shape in ['default', 'box', 'round', 'card'] : 'default';
                }
                // console.debug("jovi-colorized", this.title, this.color, this.bgcolor, this._shape);
                // result.serialize_widgets = true
                return result;
            }
        }
    },
})