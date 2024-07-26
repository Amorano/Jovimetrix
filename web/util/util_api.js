/**
 * File: util_api.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js"

export async function apiGet(url) {
    var response = await api.fetchApi(url, { cache: "no-store" })
    return await response.json()
}

export async function apiPost(url, data) {
    return api.fetchApi(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    })
}

export async function apiJovimetrix(id, cmd) {
    return api.fetchApi('/jovimetrix/message', {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            id: id,
            cmd: cmd
        }),
    })
}
