/**
 * File: config.js
 * Project: Jovimetrix
 *
 */

import { ComfyDialog, $el } from "/scripts/ui.js";
import * as util from './util.js';

var headID = document.getElementsByTagName("head")[0];
var cssNode = document.createElement('link');
cssNode.rel = 'stylesheet';
cssNode.type = 'text/css';
cssNode.href = 'extensions/Jovimetrix/jovimetrix.css';
headID.appendChild(cssNode);

const template_color_block = `
<tr>
    <td style="color: white; text-align: right; background: {{ background }}">{{ name }}</td>
    <td><input class="jov-color" type="text" name="{{ name }}" value="title" color="{{title}}" part="title"></input></td>
    <td><input class="jov-color" type="text" name="{{ name }}" value="body" color="{{body}}" part="body"></input></td>
    <td><button type="button" onclick="color_clear('{{name}}')"></button></td>
</tr>
`

export class JovimetrixConfigDialog extends ComfyDialog {

    startDrag = (e) => {
        this.dragData = {
            startX: e.clientX,
            startY: e.clientY,
            offsetX: this.element.offsetLeft,
            offsetY: this.element.offsetTop,
        };

        document.addEventListener('mousemove', this.dragMove);
        document.addEventListener('mouseup', this.dragEnd);
    }

    dragMove = (e) => {
        const { startX, startY, offsetX, offsetY } = this.dragData;
        const newLeft = offsetX + e.clientX - startX;
        const newTop = offsetY + e.clientY - startY;

        // Get the dimensions of the parent element
        const parentWidth = this.element.parentElement.clientWidth;
        const parentHeight = this.element.parentElement.clientHeight;

        // Ensure the new position is within the boundaries
        const halfX = this.element.clientWidth / 2;
        const halfY = this.element.clientHeight / 2;
        const clampedLeft = Math.max(halfX, Math.min(newLeft, parentWidth - halfX));
        const clampedTop = Math.max(halfY, Math.min(newTop, parentHeight - halfY));

        this.element.style.left = `${clampedLeft}px`;
        this.element.style.top = `${clampedTop}px`;
    }

    dragEnd = () => {
        document.removeEventListener('mousemove', this.dragMove);
        document.removeEventListener('mouseup', this.dragEnd);
    }

    createColorBlock(name, title, body) {
        return [
            $el("td", {
                style: {
                    align: "right"
                },
                textContent: name
            }),
            $el("td", [
                $el("input", {
                    class: "jov-color",
                    type: "text",
                    name: name,
                    value: "title",
                    color: title,
                    part: title
                })
            ]),
            $el("td", [
                $el("input", {
                    class: "jov-color",
                    type: "text",
                    name: name,
                    value: "body",
                    color: body,
                    part: body
                })
            ]),
            $el("td", [
                $el("button", {
                    type: "button",
                    onclick: () => {
                        color_clear(name);
                    }
                })
            ])
        ];
    }

    createColorPalettes() {
        let colorTable = null;
        const header =
            $el("div.jov-config-column", [
                $el("table", [
                    colorTable = $el("thead", [
                    ]),
                ]),
            ]);

        // now the rest which are untracked and their "categories"

        var category = [];
        const all_nodes = Object.entries(util.NODE_LIST);

        all_nodes.sort(function (a, b) {
            return a[1].category.toLowerCase().localeCompare(b[1].category.toLowerCase());
        });

        var background = ["#1A1F1A", "#1D1A1A"];
        var background_title = ["#1A271A", "#2A1A1A"];
        var background_index = 0;

        all_nodes.forEach(entry => {
            var name = entry[0];
            var data = {};

            var html = "";
            var cat = entry[1].category

            if(category.includes(cat) == false) {
                background_index = (background_index + 1) % 2;
                if (util.THEME.hasOwnProperty(cat)) {
                    data = {
                        name: cat,
                        title: util.THEME[cat].title,
                        body: util.THEME[cat].body,
                        background: background[background_index]
                    };
                } else {
                    data = {
                        name: cat,
                        title: '#4D4D4DEE',
                        body: '#4D4D4DEE',
                        background: background_title[background_index]
                    };
                }
                html = util.renderTemplate(template_color_block, data);
                colorTable.innerHTML += html;
                category.push(cat);
            }

            if (util.THEME.hasOwnProperty(name)) {
                data = {
                    name: entry[0],
                    title: util.THEME[name].title,
                    body: util.THEME[name].body,
                    background: background[background_index]
                };
            } else {
                data = {
                    name: entry[0],
                    title: '#4D4D4DEE',
                    body: '#4D4D4DEE',
                    background: background[background_index]
                };
            }
            html = util.renderTemplate(template_color_block, data);
            colorTable.innerHTML += html;
        });
        return [header];
	}

    createTitle() {
        const title = [
            "COLOR CONFIGURATION",
            "COLOR CALIBRATION",
            "COLOR CUSTOMIZATION",
            "CHROMA CALIBRATION",
            "CHROMA CONFIGURATION",
            "CHROMA CUSTOMIZATION",
            "CHROMATIC CALIBRATION",
            "CHROMATIC CONFIGURATION",
            "CHROMATIC CUSTOMIZATION",
            "HUE HOMESTEAD",
            "PALETTE PREFERENCES",
            "PALETTE PERSONALIZATION",
            "PALETTE PICKER",
            "PIGMENT PREFERENCES",
            "PIGMENT PERSONALIZATION",
            "PIGMENT PICKER",
            "SPECTRUM STYLING",
            "TINT TAILORING",
            "TINT TWEAKING"
        ];
        const randomIndex = Math.floor(Math.random() * title.length);
        return title[randomIndex];
    }

    createTitleElement() {
        return $el("table", [
            $el("tr", [
                $el("td", [
                    $el("div", [
                        this.headerTitle = $el("div.jov-title", [this.title]),
                        $el("div", [
                            $el("label", {
                                id: "jov-apply-button",
                                textContent: "FORCE NODES TO SYNCHRONIZE WITH PANEL? ",
                                style: {fontsize: "0.5em"}
                            }, [
                                $el("input", {
                                    type: "checkbox",
                                    checked: util.CONFIG_USER.color.overwrite,
                                    style: { color: "white" },
                                    onclick: (cb) => {
                                        util.CONFIG_USER.color.overwrite = cb.target.checked;
                                        var data = {
                                            id: util.USER + '.color.overwrite',
                                            v: util.CONFIG_USER.color.overwrite
                                        }
                                        util.api_post('/jovimetrix/config', data);
                                    }
                                })
                            ]),
                        ]),
                    ]),
                ]),
            ])
        ]);
    }

    createCloseButton() {
        return $el("button", {
            id: "jov-close-button",
            type: "button",
            textContent: "CLOSE",
            onclick: () => this.close()
        });
    }

    createContent() {
        const content = $el("div.comfy-modal-content", [
            this.createTitleElement(),
            $el("div.jov-config-color", [...this.createColorPalettes()]),
            this.createCloseButton()
        ]);

        content.style.width = '100%';
        content.style.height = '100%';
        content.addEventListener('mousedown', this.startDrag);
        return content;
    }

    color_clear(name) {
        var body = {
            "name": name,
        }
        util.api_post("/jovimetrix/config/clear", body);
        delete util.THEME[name];
    }

    constructor() {
        super();
        this.color_group_titleA = "#1A1F1A";
        this.color_group_backA = "#1A1F1A";
        this.color_group_titleB = "#1A271A";
        this.color_group_backB = "#2A1A1A";
        this.headerTitle = null;
        this.overwrite = false;
        this.visible = false;
        this.title = this.createTitle();
        this.element = $el("div.comfy-modal", { id:'jov-manager-dialog', parent: document.body }, [ this.createContent() ]);
    }

	show() {
        this.visible = !this.visible;
        this.headerTitle.innerText = this.createTitle();
        this.element.style.display = this.visible ? "block" : "";
    }
}

