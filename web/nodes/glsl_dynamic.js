/**
 * File: glsl_dynamic.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js";
import{ widgetSizeModeHook } from '../util/util_jov.js'

const _id = "GLSL DYNAMIC (JOV) 🧙🏽";

app.registerExtension({
    name: 'jovimetrix.node.' + _id,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData.name.endsWith("(JOV) 🧙🏽") && !nodeData.name.endsWith("(JOV) 🧙🏽‍♀️")) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            widgetSizeModeHook(this);
            return me;
        }
    }
});
