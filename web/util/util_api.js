/**
 * File: util_api.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { api } from "../../../scripts/api.js"

export async function apiGet(url) {
    var response = await api.fetchApi(url, { cache: "no-store" })
    return await response.json()
}

export async function apiJovimetrix(id, cmd, route="message") {
    try {
        const response = await api.fetchApi(`/jovimetrix/${route}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                id: id,
                cmd: cmd
            }),
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status} - ${response.statusText}`);
        }
        return response;

    } catch (error) {
        console.error("API call to Jovimetrix failed:", error);
        throw error; // or return { success: false, message: error.message }
    }
}


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
    localStorage[`Comfy.Settings.${id}`] = val;
}

export function setting_make(id, pretty, type, tip, value, attrs={}, options=[], proto=undefined) {
    const key = `JOVIMETRIX 🔺🟩🔵.${id}`
    const setting_root = `Comfy.Settings.jov.${key}`;
    const local = localStorage[setting_root];
    value = local ? local : value;

    if (proto === undefined) {
        proto = (v) => {
            apiJovimetrix(key, v, 'config');
            localStorage[setting_root] = v;
        }
    }

    app.ui.settings.addSetting({
        id: key,
        name: pretty,
        type: type,
        tooltip: tip,
        defaultValue: value,
        attrs: attrs,
        options: options,
        proto
    })
}
