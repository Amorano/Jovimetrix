/**
 * File: util_config.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { apiJovimetrix } from './util_api.js'

export async function local_get(url, d) {
    const v = localStorage.getItem(url)
    if (v && !isNaN(+v)) {
        return v;
    }
    return d;
}

export async function local_set(url, v) {
    localStorage.setItem(url, v)
}

export function setting_store(id, val) {
    apiJovimetrix(id, val, 'config');
    // CONFIG_USER[id] = val;
    localStorage[`Comfy.Settings.${id}`] = val;
}

export function setting_make(id, pretty, type, tip, value,) {
    const key = `JOVIMETRIX 🔺🟩🔵.${id}`
    const setting_root = `Comfy.Settings.jov.${key}`;
    const local = localStorage[setting_root];
    // CONFIG_USER?.[key] ? CONFIG_USER[key] :
    value = local ? local : value;

    app.ui.settings.addSetting({
        id: key,
        name: pretty,
        type: type,
        tooltip: tip,
        defaultValue: value,
        onChange(val) {
            apiJovimetrix(key, val, 'config');
            // CONFIG_USER[id] = val;
            localStorage[setting_root] = val;
        },
    })
}
