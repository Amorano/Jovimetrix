/**
 * File: util_jov.js
 * Project: Jovimetrix
 *
 */

import { fitHeight } from './util.js'
import { show_vector, widget_find, widget_hide, widget_show } from './util_widget.js'

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

export function hook_widget_AB(node, control_key) {
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

    const oldCallback = combo.callback;
    combo.callback = () => {
        oldCallback?.apply(this, arguments);
        widget_hide(node, A, "-jovi");
        widget_hide(node, B, "-jovi");
        if (["VEC2", "VEC2INT", "COORD2D", "VEC3", "VEC3INT", "VEC4", "VEC4INT", "BOOLEAN", "INT", "FLOAT"].includes(combo.value)) {
            show_vector(A, data.track_xyzw, combo.value);
            show_vector(B, data.track_yyzw, combo.value);
        }
        fitHeight(node);
    }

    setTimeout(() => { combo.callback(); }, 10);
    setCallback(A, data.track_xyzw);
    setCallback(B, data.track_yyzw);
    return data;
}

/*
const widget_x4 = this.widgets.find(w => w.name === '🅰️🅰️');
            const widget_y4 = this.widgets.find(w => w.name === '🅱️🅱️');
            widget_x4.options.menu = false;
            widget_y4.options.menu = false;
            let bool_x = {0:false}
            let bool_y = {0:false}
            let track_xyzw = {0:0, 1:0, 2:0, 3:0};
            let track_yyzw = {0:0, 1:0, 2:0, 3:0};
            const widget_combo = this.widgets.find(w => w.name === '❓');
            widget_combo.callback = () => {
                const data_x = (widget_combo.value === "BOOLEAN") ? bool_x : track_xyzw;
                const data_y = (widget_combo.value === "BOOLEAN") ? bool_y : track_yyzw;
                show_vector(widget_x4, data_x, widget_combo.value);
                show_vector(widget_y4, data_y, widget_combo.value);
                this.outputs[0].name = widget_type_name(widget_combo.value);
                fitHeight(this);
            }

            widget_x4.callback = (value) => {
                if (widget_x4.type === "toggle") {
                    bool_x[0] = value;
                } else {
                    Object.keys(widget_x4.value).forEach((key) => {
                        track_xyzw[key] = widget_x4.value[key];
                    });
                }
            }

            widget_y4.callback = () => {
                if (widget_y4.type === "toggle") {
                    bool_y[0] = widget_y4.value;
                } else {
                    Object.keys(widget_y4.value).forEach((key) => {
                        track_yyzw[key] = widget_y4.value[key];
                    });
                }
            }
            setTimeout(() => { widget_combo.callback(); }, 10);
            return me;*/