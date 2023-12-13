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
    <td style="color: white; text-align: right">{{ name }}</td>
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

        var existing = [];
        const COLORS = Object.entries(util.CONFIG.color);
        COLORS.forEach(entry => {
            existing.push(entry[0]);
            var data = {
                name: entry[0],
                title: entry[1].title,
                body: entry[1].body
            };
            const html = util.renderTemplate(template_color_block, data);
            colorTable.innerHTML += html;
        });

        // now the rest which are untracked and their "categories"
        var categories = [];
        const nodes = Object.entries(util.NODE_LIST);
        nodes.forEach(entry => {
            var name = entry[0];
            if (existing.includes(name) == false) {
                var data = {
                    name: entry[0],
                    title: '#4D4D4DEE',
                    body: '#4D4D4DEE',
                };
                const html = util.renderTemplate(template_color_block, data);
                colorTable.innerHTML += html;
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
                    title: '#3F3F3FEE',
                    body: '#3F3F3FEE',
                };
                const html = util.renderTemplate(template_color_block, data);
                colorTable.innerHTML += html;
            }
        });
		return [header];
	}

    // @TODO: re-write parser as straight $el
    createColorPalettes2() {
        var existing = [];
        const color_user = [];
        const color_node = [];
        const color_group = [];
        const COLORS = Object.entries(util.CONFIG.color);

        COLORS.forEach(entry => {
            existing.push(entry[0]);
            color_user.push(this.createColorBlock(entry[0], entry[1].title, entry[1].body));
        });

        // now the rest which are untracked and their "categories"
        var categories = [];
        const nodes = Object.entries(util.NODE_LIST);
        nodes.forEach(entry => {
            var name = entry[0];
            if (existing.includes(name) == false) {
                color_node.push(this.createColorBlock(name, '#4D4D4DEE', '#4D4D4DEE'));
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
                color_group.push(this.createColorBlock(entry[1], '#3F3F3FEE', '#3F3F3FEE'));
            }
        });

        const colorUserElements = color_user.map(row => $el("tr", row));
        const colorNodeElements = color_node.map(row => $el("tr", row));
        const colorGroupElements = color_group.map(row => $el("tr", row));

        return [
            $el("div.jov-config-column", [
                $el("table", [
                    $el("thead", [
                        $el("tr", [...colorUserElements]),
                        $el("tr", [...colorNodeElements]),
                        $el("tr", [...colorGroupElements]),
                    ]),
                ]),
            ])
        ];
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
                                    checked: this.overwrite,

                                    style: { color: "white" },
                                    onclick: (cb) => {
                                        this.overwrite = cb.target.checked;
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

    constructor() {
        super();
        this.headerTitle = null;
        this.overwrite = false;
        this.title = this.createTitle();
        this.element = $el("div.comfy-modal", { id:'jov-manager-dialog', parent: document.body }, [ this.createContent() ]);
    }

	show() {
        this.headerTitle.innerText = this.createTitle();
        this.element.style.display = "block";
    }
}
