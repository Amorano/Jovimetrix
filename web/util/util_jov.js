/**
 * File: util_jov.js
 * Project: Jovimetrix
 *
 */

import { fitHeight } from './util.js'
import { widget_hide, widget_show } from './util_widget.js'

export function hook_widget_size_mode(node) {
    const wh = node.widgets.find((w) => w.name === '🇼🇭');
    const samp = node.widgets.find((w) => w.name === '🎞️');
    const matte = node.widgets.find((w) => w.name === 'MATTE');
    const mode = node.widgets.find((w) => w.name === 'MODE');
    mode.callback = () => {
        widget_hide(node, wh);
        widget_hide(node, samp);
        widget_hide(node, matte);
        if (!['NONE'].includes(mode.value)) {
            widget_show(wh);
        }
        if (['FIT', 'ASPECT_LONG', 'ASPECT_SHORT'].includes(mode.value)) {
            widget_show(samp);
        }
        if (!['FIT'].includes(mode.value)) {
            widget_show(matte);
        }
        fitHeight(node);
    }
    setTimeout(() => { mode.callback(); }, 15);
    return mode;
}
