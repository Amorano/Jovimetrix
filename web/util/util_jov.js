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
    const mode = node.widgets.find((w) => w.name === 'MODE');
    mode.callback = () => {
        widget_hide(node, wh);
        widget_hide(node, samp);
        if (!['NONE'].includes(mode.value)) {
            widget_show(wh);
        }
        if (!['NONE', 'CROP', 'MATTE'].includes(mode.value)) {
            widget_show(samp);
        }
        fitHeight(node);
    }
    setTimeout(() => { mode.callback(); }, 10);
    return mode;
}
