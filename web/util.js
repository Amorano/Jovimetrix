import { api } from "../../../scripts/api.js";

export async function CONFIG() {
    return await api_get("/jovimetrix/config");
}

export async function NODE_LIST() {
    return await api_get("./../object_info");
}

export async function api_get(url) {
    var response = await api.fetchApi(url, { cache: "no-store" });
    return await response.json();
}

export async function api_post(url, data) {
    return api.fetchApi(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    });
}