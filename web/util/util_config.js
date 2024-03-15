/**
 * File: util_config.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { api_get } from './util_api.js'

export let NODE_LIST = await api_get("./../object_info")
export let CONFIG_CORE = await api_get("/jovimetrix/config")
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

export function setting_make(id, name, type, tip, value, callback=undefined) {
    app.ui.settings.addSetting({
        id: id,
        name: name,
        type: type,
        tooltip: tip,
        defaultValue: value,
        onChange(v) {
            if (callback !== undefined) {
                callback(v)
            }
        },
    })
}
