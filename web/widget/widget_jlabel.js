/**
 * File: widget_jlabel.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js"
import * as util from '../core/util.js'
import * as util_dom from '../core/util_dom.js'

export const jLabelWidget = (label) => {
    const widget = {
        value: label,
        type: "JLABEL",
        options: {
            serialize: false,
        }
    };

    widget.draw = function(ctx, node, widget_width, y, widget_height) {

    }

    widget.mouse = function(event, pos, node) {

    },

    widget.computeSize = function() {
        return [0, 20];
    }

    return widget;
}