/**
 * File: jovimetrix.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { ComfyDialog, $el } from "../../../scripts/ui.js";
import "./extern/colors.js";
import "./extern/colorPicker.data.js";
import "./extern/colorPicker.js";

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

function col_color(name, data, attr) {
    var col = document.createElement('td');
    var box = document.createElement('input');
    box.classList.add('color');
    if (data === undefined) {
        data = '#7F7F7F';
    }
    box.setAttribute("value", data)
    box.setAttribute("jovi", name)
    box.setAttribute("part", attr)
    col.appendChild(box);
    return col;
}

function box_color(root, name, title, body,) {
    var row = document.createElement('tr');
    var col = document.createElement('td');
    col.innerHTML = name;
    row.appendChild(col);
    row.appendChild(col_color(name, title, 'title'));
    row.appendChild(col_color(name, body, 'body'));
    root.appendChild(row);
}

// -----------
class JovimetrixConfigDialog extends ComfyDialog {
    createControlsMid() {
		let self = this;

		var headID = document.getElementsByTagName("head")[0];
        var cssNode = document.createElement('link');
        cssNode.type = 'text/css';
        cssNode.rel = 'stylesheet';
        cssNode.href = './css/style.css';
        cssNode.media = 'screen';
        headID.appendChild(cssNode);

        const NODE_LIST = api_get("./../object_info");

        const div = document.getElementById('configColor');
        if (this.CONFIG.color === undefined) {
            console.debug("💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀")
            return;
        }

        var existing = [];
        const COLORS = Object.entries(this.CONFIG.color)
        COLORS.forEach(entry => {
            box_color(div, entry[0], entry[1].title, entry[1].body);
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

        jsColorPicker('input.color', {
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

        const res =
			[
				$el("br", {}, []),


				$el("br", {}, []),
				$el("button.cm-button", {
					type: "button",
					textContent: "Alternatives of A1111",
					onclick:
						() => {
							if(!AlternativesInstaller.instance)
								AlternativesInstaller.instance = new AlternativesInstaller(app, self);
							AlternativesInstaller.instance.show();
						}
				})
			];

		return res;
	}

	createControlsLeft() {
	}

	createControlsRight() {
	}

	constructor() {
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