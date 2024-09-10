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

