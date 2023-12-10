/**
 * File: jovimetrix.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { ComfyDialog, $el } from "../../../scripts/ui.js";
import { template_colors } from './template.js'

//import "./extern/colors.js";
//import "./extern/colorPicker.data.js";
//import "./extern/colorPicker.js";

import "./extern/color.all.min.js";
//import "./extern/jsColorPicker.min.js";
import "./extern/jsColor.js";

var headID = document.getElementsByTagName("head")[0];
var cssNode = document.createElement('link');
cssNode.rel = 'stylesheet';
cssNode.type = 'text/css';
cssNode.href = 'extensions/Jovimetrix/jovimetrix.css';
headID.appendChild(cssNode);

async function api_get(url) {
    var response = await api.fetchApi(url, { cache: "no-store" });
    return await response.json();
}

async function api_post(url, data) {
    return api.fetchApi(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    });
}

const getVal = (url, d) => {
    const v = localStorage.getItem(url);
    if (v && !isNaN(+v)) {
        return v;
    }
    return d;
};

const saveVal = (url, v) => {
    localStorage.setItem(url, v);
};

export function renderTemplate(template, data) {
    // Replace placeholders in the template with corresponding data values
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
            template = template.replace(regex, data[key]);
        }
    }
    return template;
}

class JovimetrixConfigDialog extends ComfyDialog {

    createElements() {
        let html = []

        const NODE_LIST = api_get("./../object_info");

        const div = document.getElementById('configColor');
        if (jovimetrix.CONFIG.color === undefined) {
            console.debug("💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀")
            return;
        }

        var existing = [];
        const COLORS = Object.entries(jovimetrix.CONFIG.color)
        COLORS.forEach(entry => {
            renderTemplate(template_colors, entry)
            // box_color(div, entry[0], entry[1].title, entry[1].body);
            existing.push(entry[0])
        });

        // now the rest which are untracked....
        var nodes = Object.entries(NODE_LIST);

        var categories = [];
        nodes.forEach(entry => {
            var name = entry[0];
            var cat = entry[1].category;
            //console.debug(cat)
            if (existing.includes(name) == false) {
                box_color(div, entry[0], '#7F7F7F', '#7F7F7F');
            }
            if (categories.includes(cat) == false) {
                // console.debug(cat, categories)
                categories.push(cat);
            }
        });

        categories.sort(function (a, b) {
            return a.toLowerCase().localeCompare(b.toLowerCase());
        });

        Object.entries(categories).forEach(entry => {
            if (existing.includes(entry[1]) == false) {
                box_color(div, entry[1], '#3F3F3F', '#3F3F3F');
            }
        });

        window.jsColorPicker('input.color', {
            readOnly: true,
            size: 2,
            multipleInstances: false,
            //mode: 'HEX',
            noAlpha: false,
            init: function(elm, rgb) {
              elm.style.backgroundColor = elm.value;
              elm.style.color = rgb.rgbaMixCustom.luminance > 0.22 ? '#222' : '#ddd';
            },
            convertCallback: function(data, type) {

                const AHEX = options.isIE8 ? (data.alpha < 0.16 ? '0' : '') +
					(Math.round(data.alpha * 100)).toString(16).toUpperCase() + data.HEX : ''

                const name = this.patch.attributes.jovi.value;
                const part = this.patch.attributes.part.value;
                var body = {}
                body[name] = {"part": part, "color": AHEX}
                api_post("/config", body);
                COLORS[name] = data.HEX;
            },
        });

        var body = document.createElement("div");
		html.push(body);
		// init_notice(body);
		return html;
	}

    constructor() {
        super();

		const close_button = $el("button", {
            id: "jov-close-button",
            type: "button",
            textContent: "Close",
            onclick: () => this.close()
        });

		const content =
            $el("div.comfy-modal-content",
                [
                    $el("tr.jov-title", {}, [
                            $el("font", {size:6, color:"white"}, [`JOVIMETRIX CONFIGURATION`])]
                        ),
                    $el("br", {}, []),
                    $el("div.jov-menu-container",
                        [
                            $el("div.jov-menu-column", [...this.createElements()]),
                        ]),

                    $el("br", {}, []),
                    close_button,
                ]
            );

		content.style.width = '100%';
		content.style.height = '100%';
		this.element = $el("div.comfy-modal", { id:'jov-manager-dialog', parent: document.body }, [ content ]);
	}

	show() {
		this.element.style.display = "block";
	}
}

class Jovimetrix {
    constructor() {
        if (!Jovimetrix.instance) {
            Jovimetrix.instance = this;
            this.initialized = false;
        }
        return Jovimetrix.instance;
    }

    async initialize() {
        if (!this.initialized) {
            try {
                this.CONFIG = await api_get("/jovimetrix/config/raw");
                this.settings = new JovimetrixConfigDialog();
                this.initialized = true;
            } catch (error) {
                console.error('Jovimetrix failed:', error);
            }
        }
    }
}

Jovimetrix.instance = null;
export const jovimetrix = new Jovimetrix();
await jovimetrix.initialize();