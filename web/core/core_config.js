/**
 * File: core_config.js
 * Project: Jovimetrix
 *
 */

import { ComfyDialog, $el } from "../../../scripts/ui.js"
import { apiPost } from '../util/util_api.js'
import { nodeColorAll } from './core_colorize.js'
import * as util_config from '../util/util_config.js'

// Append CSS
document.head.appendChild(Object.assign(document.createElement('link'), {
    rel: 'stylesheet',
    type: 'text/css',
    href: 'extensions/Jovimetrix/jovimetrix.css'
}));

const createColorInput = (value, name, color) => $el("input.jov-color", { value, name, color });

const templateColorBlock = ({ name, background, title, body }) => (
    $el("tr", { style: { background } }, [
        $el("td", { textContent: name }),
        $el("td", [createColorInput("T", `${name}.title`, title)]),
        $el("td", [createColorInput("B", `${name}.body`, body)])
    ])
);

const templateColorHeader = ({ name, background, title, body }) => (
    $el("tr", [
        $el("td.jov-config-color-header", { style: { background }, textContent: name }),
        $el("td", [createColorInput("T", `${name}.title`, title)]),
        $el("td", [createColorInput("B", `${name}.body`, body)])
    ])
);

const updateRegexColor = (index, key, value) => {
    util_config.CONFIG_REGEX[index][key] = value;
    apiPost("/jovimetrix/config", {
        id: `${util_config.USER}.color.regex`,
        v: util_config.CONFIG_REGEX
    });
    nodeColorAll();
};

const templateColorRegex = ({ idx, name, background, title, body }) => (
    $el("tr", [
        $el("td", { style: { background } }, [
            $el("input", {
                name: `regex.${idx}`,
                value: name,
                onchange: (e) => updateRegexColor(idx, "regex", e.target.value)
            })
        ]),
        $el("td", [createColorInput("T", `regex.${idx}.title`, title)]),
        $el("td", [createColorInput("B", `regex.${idx}.body`, body)])
    ])
);

/*
const colorClear = (name) => {
    apiPost("/jovimetrix/config/clear", { name });
    delete util_config.CONFIG_THEME[name];
    if (util_config.CONFIG_COLOR.overwrite) nodeColorAll();
};
*/

export class JovimetrixConfigDialog extends ComfyDialog {
    constructor() {
        super();
        this.headerTitle = null;
        this.visible = false;
        this.element = $el("div.comfy-modal", { id: 'jov-manager-dialog', parent: document.body }, [this.createContent()]);
        this.element.addEventListener('mousedown', this.startDrag);
    }

    startDrag = (e) => {
        const { clientX, clientY } = e;
        const { offsetLeft, offsetTop } = this.element;
        this.dragData = { startX: clientX, startY: clientY, offsetX: offsetLeft, offsetY: offsetTop };
        document.addEventListener('mousemove', this.dragMove);
        document.addEventListener('mouseup', this.dragEnd);
    }

    dragMove = (e) => {
        const { startX, startY, offsetX, offsetY } = this.dragData;
        const { clientWidth: parentWidth, clientHeight: parentHeight } = this.element.parentElement;
        const { clientWidth, clientHeight } = this.element;
        const halfX = clientWidth / 2, halfY = clientHeight / 2;

        this.element.style.left = `${Math.max(halfX, Math.min(offsetX + e.clientX - startX, parentWidth - halfX))}px`;
        this.element.style.top = `${Math.max(halfY, Math.min(offsetY + e.clientY - startY, parentHeight - halfY))}px`;
    }

    dragEnd = () => {
        document.removeEventListener('mousemove', this.dragMove);
        document.removeEventListener('mouseup', this.dragEnd);
    }

    createColorPalettes() {
        const colorTable = $el("thead");
        const header = $el("div.jov-config-column", [$el("table", [colorTable])]);

        util_config.CONFIG_COLOR.regex?.forEach((entry, idx) => {
            colorTable.appendChild($el("tbody", templateColorRegex({
                idx,
                name: entry.regex,
                title: entry.title,
                body: entry.body,
                background: LiteGraph.WIDGET_BGCOLOR
            })));
        });

        const category = [];
        const all_nodes = Object.entries(util_config.NODE_LIST || []).sort((a, b) => {
            const catA = a[1].category, catB = b[1].category;
            if (catA === "" || catA.startsWith("_")) return 1;
            if (catB === "" || catB.startsWith("_")) return -1;
            return catA.toLowerCase().localeCompare(catB.toLowerCase());
        });

        const { backA, backB, titleA, titleB } = util_config.CONFIG_COLOR;
        const backgrounds = [backA, backB];
        const backgroundTitles = [titleA, titleB];
        let backgroundIndex = 0;

        all_nodes.forEach(([name, { category: cat }]) => {
            const meow = cat.split('/')[0];
            if (!category.includes(meow)) {
                backgroundIndex = (backgroundIndex + 1) % 2;
                colorTable.appendChild($el("tbody", templateColorHeader({
                    name: meow,
                    background: LiteGraph.WIDGET_BGCOLOR,
                    ...util_config.CONFIG_THEME[meow]
                })));
                category.push(meow);
            }

            if (!category.includes(cat)) {
                backgroundIndex = (backgroundIndex + 1) % 2;
                colorTable.appendChild($el("tbody", templateColorHeader({
                    name: cat,
                    background: backgroundTitles[backgroundIndex] || LiteGraph.WIDGET_BGCOLOR,
                    ...util_config.CONFIG_THEME[cat]
                })));
                category.push(cat);
            }

            colorTable.appendChild($el("tbody", templateColorBlock({
                name,
                ...util_config.CONFIG_THEME[name],
                background: backgrounds[backgroundIndex] || LiteGraph.NODE_DEFAULT_COLOR
            })));
        });

        return [header];
    }

    createTitle = () => [
        "COLOR CONFIGURATION", "COLOR CALIBRATION", "COLOR CUSTOMIZATION",
        "CHROMA CALIBRATION", "CHROMA CONFIGURATION", "CHROMA CUSTOMIZATION",
        "CHROMATIC CALIBRATION", "CHROMATIC CONFIGURATION", "CHROMATIC CUSTOMIZATION",
        "HUE HOMESTEAD", "PALETTE PREFERENCES", "PALETTE PERSONALIZATION",
        "PALETTE PICKER", "PIGMENT PREFERENCES", "PIGMENT PERSONALIZATION",
        "PIGMENT PICKER", "SPECTRUM STYLING", "TINT TAILORING", "TINT TWEAKING"
    ][Math.floor(Math.random() * 19)];

    createTitleElement() {
        return $el("table", [$el("tr", [$el("td", [$el("div.jov-title", [
            this.headerTitle = $el("div.jov-title-header"),
            $el("label", { id: "jov-apply-button", textContent: "FORCE NODES TO SYNCHRONIZE WITH PANEL? ", style: { fontsize: "0.5em" } }, [
                $el("input", {
                    type: "checkbox",
                    checked: util_config.CONFIG_USER.color.overwrite,
                    style: { color: "white" },
                    onclick: (cb) => {
                        util_config.CONFIG_USER.color.overwrite = cb.target.checked;
                        apiPost('/jovimetrix/config', {
                            id: `${util_config.USER}.color.overwrite`,
                            v: util_config.CONFIG_USER.color.overwrite
                        });
                        if (util_config.CONFIG_USER.color.overwrite) nodeColorAll();
                    }
                })
            ])
        ])])])]);
    }

    createContent() {
        return $el("div.comfy-modal-content", {
            style: { width: '100%', height: '100%' }
        }, [
            this.createTitleElement(),
            $el("div.jov-config-color", [...this.createColorPalettes()]),
            $el("button", {
                id: "jov-close-button",
                type: "button",
                textContent: "CLOSE",
                onclick: () => {
                    this.close();
                    this.visible = false;
                }
            })
        ]);
    }

    show() {
        this.visible = !this.visible;
        this.headerTitle.innerText = this.createTitle();
        this.element.style.display = this.visible ? "block" : "";
    }
}