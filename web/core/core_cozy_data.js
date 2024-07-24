/**
 * File: core_cozy_data.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"

const JDataBucket = (app, name, opts) => {
    let options = opts || {};
    options.serialize = false;
    const w = {
        name: name,
        type: "JDATABUCKET",
        hidden: true,
        options: options,
        draw: function (ctx, node, width, Y, height) {
            return;
        },
        computeSize: function (width) {
            return [0, 0];
        }
    }
    return w
}

app.registerExtension({
    name: "jovimetrix.data.bucket",
    async getCustomWidgets(app) {
        return {
            JDATABUCKET: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(JDataBucket(app, inputName, inputData[1]))
            })
        }
    }
});