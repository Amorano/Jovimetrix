/**
 * File: util_jov.js
 * Project: Jovimetrix
 *
 */

import { nodeFitHeight } from './util_node.js'
import { widgetShowVector, widget_type_name, widgetHide, widgetShow } from './util_widget.js'

export function widgetSizeModeHook(node, wh_hide=true) {
    const wh = node.widgets.find(w => w.name === '🇼🇭');
    const samp = node.widgets.find(w => w.name === '🎞️');
    const mode = node.widgets.find(w => w.name === 'MODE');
    mode.callback = () => {
        if (wh_hide) {
            widgetHide(node, wh);
            widgetHide(node, samp);

            if (!['NONE'].includes(mode.value)) {
                widgetShow(wh);
            }
            if (!['NONE', 'CROP', 'MATTE'].includes(mode.value)) {
                widgetShow(samp);
            }
            nodeFitHeight(node);
        }
    }
    setTimeout(() => { mode.callback(); }, 20);
}

export function widgetSizeModeHook2(nodeType, wh_hide=true) {
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function (node) {
        const me = onNodeCreated?.apply(this);
        const wh = node.widgets.find(w => w.name === '🇼🇭');
        const samp = node.widgets.find(w => w.name === '🎞️');
        const mode = node.widgets.find(w => w.name === 'MODE');
        mode.callback = () => {
            if (wh_hide) {
                widgetHide(node, wh);
            }
            widgetHide(node, samp);
            if (!['NONE'].includes(mode.value)) {
                widgetShow(wh);
            }
            if (!['NONE', 'CROP', 'MATTE'].includes(mode.value)) {
                widgetShow(samp);
            }
            nodeFitHeight(node);
        }
        setTimeout(() => { mode.callback(); }, 20);
        return me;
    }
}

export function widgetOutputHookType(node, control_key, match_output=0) {
    const combo = node.widgets.find(w => w.name === control_key);
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

/*
* matchFloatSize forces the target to be float[n] based on its type size
*/
export function widgetHookAB(node, control_key) {

    const AA = node.widgets.find(w => w.name === '🅰️🅰️');
    const BB = node.widgets.find(w => w.name === '🅱️🅱️');
    const combo = node.widgets.find(w => w.name === control_key);

    if (combo === undefined) {
        return;
    }

    widgetHookControl(node, control_key, AA);
    widgetHookControl(node, control_key, BB);
    widgetOutputHookType(node, control_key);
    setTimeout(() => { combo.callback(); }, 5);

    return combo;
};

/*
* matchFloatSize forces the target to be float[n] based on its type size
*/
export function widgetHookControl(node, control_key, target, matchFloatSize=false) {

    const initializeTrack = (widget) => {
        const track = {};
        for (let i = 0; i < 4; i++) {
            track[i] = widget.options?.default[i];
        }
        Object.assign(track, widget.value);
        return track;
    };

    const setCallback = (widget, trackKey) => {
        widget.options.menu = false;
        widget.callback = () => {
            if (widget.type === "toggle") {
                trackKey[0] = widget.value ? 1 : 0;
            } else {
                Object.keys(widget.value).forEach((key) => {
                    trackKey[key] = widget.value[key];
                });
            }
        };
    };

    const { widgets } = node;
    const combo = widgets.find(w => w.name === control_key);

    if (!target || !combo) {
        throw new Error("Required widgets not found");
    }

    const data = {
        track_xyzw: initializeTrack(target),
        target,
        combo
    };

    const oldCallback = combo.callback;
    combo.callback = () => {
        const me = oldCallback?.apply(this, arguments);
        widgetHide(node, target, "-jov");
        if (["VEC2", "VEC2INT", "COORD2D", "VEC3", "VEC3INT", "VEC4", "VEC4INT", "BOOLEAN", "INT", "FLOAT"].includes(combo.value)) {
            let size = combo.value;
            if (matchFloatSize) {
                size = "FLOAT";
                if (["VEC2", "VEC2INT", "COORD2D"].includes(combo.value)) {
                    size = "VEC2";
                } else if (["VEC3", "VEC3INT"].includes(combo.value)) {
                    size = "VEC3";
                } else if (["VEC4", "VEC4INT"].includes(combo.value)) {
                    size = "VEC4";
                }
            }
            widgetShowVector(target, data.track_xyzw, size);
        }
        nodeFitHeight(node);
        return me;
    }
    setCallback(target, data.track_xyzw);
    return data;
}
