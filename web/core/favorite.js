/**
 * File: favorite.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from './util.js'

const ext = {
    name: "jovimetrix.favorites",
    async init(app) {


    },
    async setup(app) {

    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {

        }
    },
}

app.registerExtension(ext)
