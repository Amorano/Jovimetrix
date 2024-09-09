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
    try {
        const response = await api.fetchApi('/jovimetrix/message', {
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
        console.debug(response);
        return response;

    } catch (error) {
        console.error("API call to Jovimetrix failed:", error);
        throw error; // or return { success: false, message: error.message }
    }
}

