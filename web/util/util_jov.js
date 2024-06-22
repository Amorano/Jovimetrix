/**
 * File: util_jov.js
 * Project: Jovimetrix
 *
 */

import { fitHeight } from './util.js'
import { widget_type_name, show_vector, widget_find, widget_hide, widget_show } from './util_widget.js'

export function hook_widget_size_mode(node, wh_hide=true) {
    const wh = widget_find(node.widgets, '🇼🇭');
    const samp = widget_find(node.widgets, '🎞️');
    const mode = widget_find(node.widgets, 'MODE');
    mode.callback = () => {
        if (wh_hide) {
            widget_hide(node, wh);
        }
        widget_hide(node, samp);
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

export function hook_widget_AB2(node) {
    const data = {
        bool_x: {0:A.value[0] != 0},
        bool_y: {0:B.value[0] != 0},
        track_xyzw: initializeTrack(A),
        track_yyzw: initializeTrack(B),
        A: node.widgets.find(w => w.name === '🅰️🅰️'),
        B: node.widgets.find(w => w.name === '🅱️🅱️')
    }
    data.A.options.menu = false;
    data.B.options.menu = false;
    data.A.callback = () => {
        if (data.A.type === "toggle") {
            data.bool_x[0] = data.A.value;
        } else {
            Object.keys(data.A.value).forEach((key) => {
                data.track_xyzw[key] = data.A.value[key];
            });
        }
    }
    data.B.callback = () => {
        if (data.B.type === "toggle") {
            data.bool_y[0] = data.B.value;
        } else {
            Object.keys(data.B.value).forEach((key) => {
                data.track_yyzw[key] = data.B.value[key];
            });
        }
    }
    return data;
}

export function hook_widget_AB(node, control_key) {
    const initializeBool = (value) => ({ 0: value[0] !== 0 });
    const initializeTrack = (widget) => {
        const track = {};
        for (let i = 0; i < 4; i++) {
            track[i] = widget.options.default[i];
        }
        Object.assign(track, widget.value);
        return track;
    };

    const setCallback = (widget, boolKey, trackKey) => {
        widget.options.menu = false;
        widget.callback = () => {
            if (widget.type === "toggle") {
                boolKey[0] = widget.value;
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
        bool_x: initializeBool(A.value),
        bool_y: initializeBool(B.value),
        track_xyzw: initializeTrack(A),
        track_yyzw: initializeTrack(B),
        A,
        B,
        combo
    };

    combo.callback = () => {
        const data_x = (combo.value === "BOOLEAN") ? data.bool_x : data.track_xyzw;
        const data_y = (combo.value === "BOOLEAN") ? data.bool_y : data.track_yyzw;
        show_vector(A, data_x, combo.value);
        show_vector(B, data_y, combo.value);
        fitHeight(node);
    }

    setTimeout(() => { combo.callback(); }, 10);
    setCallback(A, data.bool_x, data.track_xyzw);
    setCallback(B, data.bool_y, data.track_yyzw);
    return data;
}