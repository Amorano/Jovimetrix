/**
 * File: util.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

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

export async function local_get(url, d) {
    const v = localStorage.getItem(url);
    if (v && !isNaN(+v)) {
        return v;
    }
    return d;
};

export async function local_set(url, v) {
    localStorage.setItem(url, v);
};

export let NODE_LIST = await api_get("./../object_info");
export let CONFIG_CORE = await api_get("/jovimetrix/config");
export let CONFIG_USER = CONFIG_CORE.user.default;
export let CONFIG_COLOR = CONFIG_USER.color;
export let THEME = CONFIG_COLOR.theme;
export let USER = 'user.default';

// gets the CONFIG entry for name
export function node_color_get(find_me) {
    let node = THEME[find_me];
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
            const found = THEME[k];
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
    app.graph._nodes.forEach((node) => {
        this.node_color_reset(node, false);
    });
    app.graph.setDirtyCanvas(true, true);
    // console.info("JOVI] all nodes color refreshed");
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
    return "#13171DFF";
}

export function toggleFoldable(elementId, symbolId) {
  const content = document.getElementById(elementId)
  const symbol = document.getElementById(symbolId)
  if (content.style.display === 'none' || content.style.display === '') {
    content.style.display = 'flex'
    symbol.innerHTML = '&#9661;' // Down arrow
  } else {
    content.style.display = 'none'
    symbol.innerHTML = '&#9655;' // Right arrow
  }
}

