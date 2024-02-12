/**
 * File: util_jov.js
 * Project: Jovimetrix
 *
 */

import { widget_show, fitHeight, widget_hide } from './util.js'

export function hook_widget_size_mode(node) {
    const wh = node.widgets.find((w) => w.name === '🇼🇭');
    const samp = node.widgets.find((w) => w.name === '🎞️');
    //const matte = node.widgets.find((w) => w.name === 'MATTE');
    const mode = node.widgets.find((w) => w.name === 'MODE');
    mode.callback = () => {
        widget_hide(node, wh);
        widget_hide(node, samp);
        //widget_hide(node, matte);
        if (!['NONE'].includes(mode.value)) {
            widget_show(wh);
        }
        if (['FIT', 'ASPECT', 'ASPECT_SHORT'].includes(mode.value)) {
            widget_show(samp);
        }
        fitHeight(node);
    }
    setTimeout(() => { mode.callback(); }, 15);
    return mode;
}
