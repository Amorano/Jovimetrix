/**
 * File: jovimetrix.js
 * Project: Jovimetrix
 *
 */

import { api } from "../../../scripts/api.js";
import { ComfyDialog, $el } from "../../../scripts/ui.js";
import { template_color_block } from './template.js'
import * as util from './util.js';
import * as coloris from './extern/coloris.min.js'

var headID = document.getElementsByTagName("head")[0];
var cssNode = document.createElement('link');
cssNode.rel = 'stylesheet';
cssNode.type = 'text/css';
cssNode.href = 'extensions/Jovimetrix/jovimetrix.css';
headID.appendChild(cssNode);
cssNode.href = 'extensions/Jovimetrix/extern/coloris.min.css';
headID.appendChild(cssNode);

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

export let jovimetrix = null;

class JovimetrixConfigDialog extends ComfyDialog {

    createElements(CONFIG, NODE_LIST) {
        let colorTable = null;
        const header =
            $el("div.tg-wrap", [
                $el("div.jov-menu-column", {
                    style: {
                        maxHeight: '634px',
                        overflowY: 'auto'
                    }}, [
                        $el("table", [
                            colorTable = $el("thead", [
                            ]),
                        ]),
                    ]),
                ]);

        var existing = [];
        const COLORS = Object.entries(CONFIG.color);
        COLORS.forEach(entry => {
            existing.push(entry[0]);
            var data = {
                name: entry[0],
                title: entry[1].title,
                body: entry[1].body,
            };
            colorTable.innerHTML += renderTemplate(template_color_block, data);
        });

        // now the rest which are untracked and their "categories"
        var categories = [];
        const nodes = Object.entries(NODE_LIST);
        nodes.forEach(entry => {
            var name = entry[0];
            if (existing.includes(name) == false) {
                var data = {
                    name: entry[0],
                    title: '#7F7F7F',
                    body: '#7F7F7F',
                };
                colorTable.innerHTML += renderTemplate(template_color_block, data);
            }

            var cat = entry[1].category;
            if (categories.includes(cat) == false) {
                categories.push(cat);
            }
        });

        categories.sort(function (a, b) {
            return a.toLowerCase().localeCompare(b.toLowerCase());
        });

        Object.entries(categories).forEach(entry => {
            if (existing.includes(entry[1]) == false) {
                var data = {
                    name: entry[1],
                    title: '#3F3F3F',
                    body: '#3F3F3F',
                };
                colorTable.innerHTML += renderTemplate(template_color_block, data);
            }
        });
		return [header];
	}

    constructor() {
        super();
        const close_button = $el("button", {
            id: "jov-close-button",
            type: "button",
            textContent: "CLOSE",
            onclick: () => this.close()
        });

        const init = async () => {
            const CONFIG = await util.CONFIG();
            const NODE_LIST = await util.NODE_LIST();

            const content =
                $el("div.comfy-modal-content",
                    [
                        $el("tr.jov-title", {}, [
                                $el("font", {size:5, color:"white"}, [`JOVIMETRIX COLOR CONFIGURATION`])]
                            ),
                        $el("br", {}, []),
                        $el("div.jov-menu-container",
                            [
                                $el("div.jov-menu-column", [...this.createElements(CONFIG, NODE_LIST)]),
                            ]),
                        $el("br", {}, []),
                        close_button,
                    ]
                );

            content.style.width = '100%';
            content.style.height = '100%';
            this.element = $el("div.comfy-modal", { id:'jov-manager-dialog', parent: document.body }, [ content ]);
        };
        init();
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
                this.settings = new JovimetrixConfigDialog();
                this.initialized = true;
            } catch (error) {
                console.error('Jovimetrix failed:', error);
            }
        }
    }
}

Jovimetrix.instance = null;
jovimetrix = new Jovimetrix();
await jovimetrix.initialize();

Coloris({
    themeMode: 'dark',
    alpha: true
  });
