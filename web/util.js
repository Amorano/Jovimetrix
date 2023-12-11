import { api } from "../../../scripts/api.js";

export function convert_hex(color) {
    if (!color.HEX.includes("NAN")) {
        return '#' + color.HEX + ((color.alpha * 255) | 1 << 8).toString(16).slice(1).toUpperCase();
    }
    return "#7F7F7FFF";
}

export const local_get = (url, d) => {
    const v = localStorage.getItem('jovi.' + url);
    if (v && !isNaN(+v)) {
        //console.info('get', 'jovi.' + url, v);
        return v;
    }
    //console.info('get', 'jovi.' + url, d);
    return d;
};

export const local_set = (url, v) => {
    localStorage.setItem('jovi.' + url, v);
    //console.info('set', 'jovi.' + url, v);
};

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