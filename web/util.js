import { api } from "../../../scripts/api.js";

async function _CONFIG() {
    return await api_get("/jovimetrix/config");
}

async function _NODE_LIST() {
    return await api_get("./../object_info");
}

export const CONFIG = await api_get("/jovimetrix/config");
export const NODE_LIST = await api_get("./../object_info");

// gets the CONFIG entry for this Node.type || Node.name
export function node_color_get(find_me) {
    let node = CONFIG.color[find_me];
    if (node) {
        return node;
    }
    node = NODE_LIST[find_me];
    //console.info(node);
    if (node && node.category) {
        //console.info(CONFIG);
        const segments = node.category.split('/');
        let k = segments.join('/');
        while (k) {
            const found = CONFIG.color[k];
            if (found) {
                //console.info(found, node.category);
                return found;
            }
            const last = k.lastIndexOf('/');
            k = last !== -1 ? k.substring(0, last) : '';
        }
    }
}

// refresh the color of a node
export function node_color_reset(node, refresh=true) {
    const data = node_color_get(node.type || node.name);
    if (data) {
        node.bgcolor = data.body;
        node.color = data.title;
        // console.info(node, data);
        if (refresh) {
            node.setDirtyCanvas(true, true);
        }
    }
}

export function node_color_list(nodes) {
    Object.entries(nodes).forEach((node) => {
        node_color_reset(node, false);
    });
    app.graph.setDirtyCanvas(true, true);
}

export function node_color_all() {
    return node_color_list(app.graph._nodes);
}

export function renderTemplate(template, data) {
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
            template = template.replace(regex, data[key]);
        }
    }
    return template;
}

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
