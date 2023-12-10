/**
 * File: jovimetrix.js
 * Project: Jovimetrix
 *
 */

import { ComfyDialog, $el } from "../../../scripts/ui.js";
import { template_color_block } from './template.js'
import * as util from './util.js';
import './extern/coloris.min.js'

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
                body: entry[1].body
            };
            var html = renderTemplate(template_color_block, data);
            colorTable.innerHTML += html;
            //console.log(colorTable.innerHTML )
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
                $el("div.comfy-modal-content", {id: "jov-manager-dialog"},
                    [
                        $el("tr", [
                                $el("font", {size:7, color:"white"}, [`JOVIMETRIX COLOR CONFIGURATION`])]
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
        this.settings = new JovimetrixConfigDialog();
    }
}

jovimetrix = new Jovimetrix();

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

let swatches_max = 16;
let swatches = [
    '#264653EE',
    '#2a9d8f',
    '#e9c46aCC',
    '#e76f51',
    '#d62828',
    '#0096c7',
    '#00b4d880',
    '#264653EE',
    '#2a9d8f',
    '#e9c46aCC',
    '#e76f51',
    '#d62828',
    '#0096c7',
    '#00b4d880',
    '#264653EE',
    '#2a9d8f'
  ]

Coloris({
    theme: 'large',
    themeMode: 'dark',
    forceAlpha: true,
    closeButton: true,
    //formatToggle: true,
    selectInput: true,
    swatches: swatches,
      onChange: (color, input) => {

      }
  });


const CONFIG = await util.CONFIG();

document.addEventListener('coloris:pick', event => {
    const name = event.detail.currentEl.name;
    const part = event.detail.currentEl.part[0].toLowerCase();
    const color = event.detail.color.toUpperCase();
    const current = CONFIG.color[name];

    if (current && current == color) {
        return;
    }

    var body = {
        "name": name,
        "part": part,
        "color": color
    }
    util.api_post("/jovimetrix/config", body);
    CONFIG.color[name] = color;
});