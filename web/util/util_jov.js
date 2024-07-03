/**
 * File: util_jov.js
 * Project: Jovimetrix
 *
 */

import { fitHeight } from './util.js'
import { show_vector, widget_find, widget_type_name, widget_find_output, widget_hide, widget_show } from './util_widget.js'

export function hook_widget_size_mode(node, wh_hide=true) {
    const wh = widget_find(node.widgets, '🇼🇭');
    const samp = widget_find(node.widgets, '🎞️');
    const mode = widget_find(node.widgets, 'MODE');
    mode.callback = () => {
        if (wh_hide) {
            widget_hide(node, wh, "-jov");
        }
        widget_hide(node, samp, "-jov");
        if (!['NONE'].includes(mode.value)) {
            widget_show(wh);
        }
        if (!['NONE', 'CROP', 'MATTE'].includes(mode.value)) {
            widget_show(samp);
        }
        fitHeight(node);
    }
    setTimeout(() => { mode.callback(); }, 20);
}

export function hook_widget_size_mode2(nodeType, wh_hide=true) {
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this);
        const wh = widget_find(node.widgets, '🇼🇭');
        const samp = widget_find(node.widgets, '🎞️');
        const mode = widget_find(node.widgets, 'MODE');
        mode.callback = () => {
            if (wh_hide) {
                widget_hide(node, wh, "-jov");
            }
            widget_hide(node, samp, "-jov");
            if (!['NONE'].includes(mode.value)) {
                widget_show(wh);
            }
            if (!['NONE', 'CROP', 'MATTE'].includes(mode.value)) {
                widget_show(samp);
            }
            fitHeight(node);
        }
        setTimeout(() => { mode.callback(); }, 20);
        return me;
    }
}

export function hook_widget_type(node, control_key, match_output=0) {
    const combo = widget_find(node.widgets, control_key);
    const output = node.outputs[match_output];

    if (!output || !combo) {
        throw new Error("Required widgets not found");
    }

    const oldCallback = combo.callback;
    combo.callback = () => {
        const me = oldCallback?.apply(this, arguments);
        node.outputs[match_output].name = widget_type_name(combo.value);
        node.outputs[match_output].type = combo.value;
        return me;
    }
    setTimeout(() => { combo.callback(); }, 10);
}

export function hook_widget_AB(node, control_key, match_output=-1) {
    const initializeTrack = (widget) => {
        const track = {};
        for (let i = 0; i < 4; i++) {
            track[i] = widget.options.default[i];
        }
        Object.assign(track, widget.value);
        return track;
    };

    const setCallback = (widget, trackKey) => {
        widget.options.menu = false;
        widget.callback = () => {
            if (widget.type === "toggle") {
                trackKey[0] = 1 ? widget.value : 0;
            } else {
                Object.keys(widget.value).forEach((key) => {
                    trackKey[key] = widget.value[key];
                });
            }
        };
    };

    const { widgets } = node;
    const A = widget_find(widgets, '🅰️🅰️');
    const B = widget_find(widgets, '🅱️🅱️');
    const combo = widget_find(widgets, control_key);

    if (!A || !B || !combo) {
        throw new Error("Required widgets not found");
    }

    const data = {
        track_xyzw: initializeTrack(A),
        track_yyzw: initializeTrack(B),
        A,
        B,
        combo
    };

    if (match_output > -1) {
        hook_widget_type(node, control_key, match_output);
    }

    const oldCallback = combo.callback;
    combo.callback = () => {
        const me = oldCallback?.apply(this, arguments);
        widget_hide(node, A, "-jovi");
        widget_hide(node, B, "-jovi");
        if (["VEC2", "VEC2INT", "COORD2D", "VEC3", "VEC3INT", "VEC4", "VEC4INT", "BOOLEAN", "INT", "FLOAT"].includes(combo.value)) {
            show_vector(A, data.track_xyzw, combo.value);
            show_vector(B, data.track_yyzw, combo.value);
        }
        fitHeight(node);
        return me;
    }

    setTimeout(() => { combo.callback(); }, 10);
    setCallback(A, data.track_xyzw);
    setCallback(B, data.track_yyzw);
    return data;
}
