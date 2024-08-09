/**
 * File: util_config.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { apiGet, apiPost } from './util_api.js'

export let NODE_LIST = await apiGet("/object_info");
export let CONFIG_CORE = await apiGet("/jovimetrix/config")
export let CONFIG_USER = CONFIG_CORE.user.default
export let CONFIG_COLOR = CONFIG_USER.color
export let CONFIG_REGEX = CONFIG_COLOR.regex || []
export let CONFIG_THEME = CONFIG_COLOR.theme
export let USER = 'user.default'

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
    var data = { id: id, v: val }
    apiPost('/jovimetrix/config', data);
    CONFIG_USER[id] = val;
    localStorage[`Comfy.Settings.${id}`] = val;
}

export function setting_make(id, pretty, type, tip, value,) {
    const key = `JOVIMETRIX 🔺🟩🔵.${id}`
    const setting_root = `Comfy.Settings.jov.${key}`;
    const local = localStorage[setting_root];
    value = local ? local : CONFIG_USER?.[key] ? CONFIG_USER[key] : value;

    app.ui.settings.addSetting({
        id: key,
        name: pretty,
        type: type,
        tooltip: tip,
        defaultValue: value,
        onChange(val) {
            var data = { id: key, v: val }
            apiPost('/jovimetrix/config', data);
            CONFIG_USER[id] = val;
            localStorage[setting_root] = val;
        },
    })
}
