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

export function setting_make(category, name, type, tip, value, attrs={}, options=[], proto=undefined) {
    const key = `JOVIMETRIX 🔺🟩🔵.${category}.${name}`;
    //const setting_root = `Comfy.Settings.jov.${key}`;
    const local = localStorage.getItem(key);
    value = local ? local : value;

    app.ui.settings.addSetting({
        id: key,
        category: ["JOVIMETRIX 🔺🟩🔵", category, name],
        name: name,
        type: type,
        tooltip: tip,
        defaultValue: value,
        attrs: attrs,
        options: options,
        async onChange(value) {
            if (proto) {
                proto(value);
            }
            apiJovimetrix(key, value, 'config');
            localStorage[key] = value;
        }
    });

}
