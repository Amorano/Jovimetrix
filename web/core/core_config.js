/**
 * File: core_config.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { ComfyDialog, $el } from "/scripts/ui.js"
import { api_post } from '../util/util_api.js'
import { node_color_all } from '../util/util_color.js'
import * as util_config from '../util/util_config.js'

var headID = document.getElementsByTagName("head")[0]
var cssNode = document.createElement('link')
cssNode.rel = 'stylesheet'
cssNode.type = 'text/css'
cssNode.href = 'extensions/Jovimetrix/jovimetrix.css'
headID.appendChild(cssNode)

const templateColorBlock = (data) => [
    $el("tr", { style: {
            background: data.background
        }}, [
            $el("td", {
                textContent: data.name
            }),
            $el("td", [
                $el("input.jov-color", {
                    value: "T",
                    name: data.name + '.title',
                    color: data.title
                })
            ]),
            $el("td", [
                $el("input.jov-color", {
                    value: "B",
                    name: data.name + '.body',
                    color: data.body
                })
            ])
        ])
]

const templateColorHeader = (data) => [
    $el("tr", [
        $el("td.jov-config-color-header", {
            style: {
                background: data.background
            },
            textContent: data.name
        }),
        $el("td", [
            $el("input.jov-color", {
                value: "T",
                name: data.name + '.title',
                color: data.title
            })
        ]),
        $el("td", [
            $el("input.jov-color", {
                value: "B",
                name: data.name + '.body',
                color: data.body
            })
        ])
    ])
]

function updateRegexColor(index, key, value) {
    util_config.CONFIG_REGEX[index][key] = value
    let api_packet = {
        id: util_config.USER + '.color.regex',
        v: util_config.CONFIG_REGEX
    }
    api_post("/jovimetrix/config", api_packet)
    node_color_all()
};

const templateColorRegex = (data) => [
    $el("tr", [
        $el("td", {
            style: {
                background: data.background
            }
        }, [
            $el("input", {
                name: "regex." + data.idx,
                value: data.name,
                onchange: function() {
                    updateRegexColor(data.idx, "regex", this.value)
                }
            })
        ]),
        $el("td", [
            $el("input.jov-color", {
                value: "T",
                name: "regex." + data.idx + ".title",
                color: data.title
            })
        ]),
        $el("td", [
            $el("input.jov-color", {
                value: "B",
                name: "regex." + data.idx + ".body",
                color: data.body
            })
        ])
    ])
]

function colorClear(name) {
    api_post("/jovimetrix/config/clear", { "name": name })
    delete util_config.CONFIG_THEME[name]
    if (util_config.CONFIG_COLOR.overwrite) {
        node_color_all()
    }
}

export class JovimetrixConfigDialog extends ComfyDialog {
    startDrag = (e) => {
        this.dragData = {
            startX: e.clientX,
            startY: e.clientY,
            offsetX: this.element.offsetLeft,
            offsetY: this.element.offsetTop,
        }

        document.addEventListener('mousemove', this.dragMove)
        document.addEventListener('mouseup', this.dragEnd)
    }

    dragMove = (e) => {
        const { startX, startY, offsetX, offsetY } = this.dragData
        const newLeft = offsetX + e.clientX - startX
        const newTop = offsetY + e.clientY - startY

        // Get the dimensions of the parent element
        const parentWidth = this.element.parentElement.clientWidth
        const parentHeight = this.element.parentElement.clientHeight

        // Ensure the new position is within the boundaries
        const halfX = this.element.clientWidth / 2
        const halfY = this.element.clientHeight / 2
        const clampedLeft = Math.max(halfX, Math.min(newLeft, parentWidth - halfX))
        const clampedTop = Math.max(halfY, Math.min(newTop, parentHeight - halfY))

        this.element.style.left = `${clampedLeft}px`
        this.element.style.top = `${clampedTop}px`
    }

    dragEnd = () => {
        document.removeEventListener('mousemove', this.dragMove)
        document.removeEventListener('mouseup', this.dragEnd)
    }

    createColorPalettes() {
        var data = {}
        let colorTable = null
        const header =
            $el("div.jov-config-column", [
                $el("table", [
                    colorTable = $el("thead", [
                    ]),
                ]),
            ])

        // rule-sets first
        var idx = 0
        const rules = util_config.CONFIG_COLOR.regex || []
        rules.forEach(entry => {
            const data = {
                idx: idx,
                name: entry.regex,
                title: entry.title, // || LiteGraph.NODE_TITLE_COLOR,
                body: entry.body, // || LiteGraph.NODE_DEFAULT_COLOR,
                background: LiteGraph.WIDGET_BGCOLOR
            }
            const row = templateColorRegex(data);
            colorTable.appendChild($el("tbody", row))
            idx += 1
        })

        // get categories to generate on the fly
        const category = []
        const all_nodes = Object.entries(util_config.NODE_LIST);
        all_nodes.sort((a, b) => {
            const categoryComparison = a[1].category.toLowerCase().localeCompare(b[1].category.toLowerCase());
            // Move items with empty category or starting with underscore to the end
            if (a[1].category === "" || a[1].category.startsWith("_")) {
                return 1;
            } else if (b[1].category === "" || b[1].category.startsWith("_")) {
                return -1;
            } else {
                return categoryComparison;
            }
        });

        // groups + nodes
        const alts = util_config.CONFIG_COLOR
        const background = [alts.backA, alts.backB]
        const background_title = [alts.titleA, alts.titleB]
        let background_index = 0
        all_nodes.forEach(entry => {
            var name = entry[0]
            var cat = entry[1].category
            var meow = cat.split('/')[0]

            if (!category.includes(meow))
            {
                // major category first?
                background_index = (background_index + 1) % 2
                data = {
                    name: meow,
                    background: LiteGraph.WIDGET_BGCOLOR
                }
                if (util_config.CONFIG_THEME.hasOwnProperty(meow)) {
                    data.title = util_config.CONFIG_THEME[meow].title
                    data.body = util_config.CONFIG_THEME[meow].body
                }
                colorTable.appendChild($el("tbody", templateColorHeader(data)))
                category.push(meow)
            }

            if(category.includes(cat) == false) {
                background_index = (background_index + 1) % 2
                data = {
                    name: cat,
                    background: background_title[background_index] || LiteGraph.WIDGET_BGCOLOR
                }
                if (util_config.CONFIG_THEME.hasOwnProperty(cat)) {
                    data.title = util_config.CONFIG_THEME[cat].title
                    data.body = util_config.CONFIG_THEME[cat].body
                }
                colorTable.appendChild($el("tbody", templateColorHeader(data)))
                category.push(cat)
            }

            const who = util_config.CONFIG_THEME[name] || {};
            data = {
                name: name,
                title: who.title,
                body: who.body,
                background: background[background_index] || LiteGraph.NODE_DEFAULT_COLOR
            }
            colorTable.appendChild($el("tbody", templateColorBlock(data)))
        })
        return [header]
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
        ]
        const randomIndex = Math.floor(Math.random() * title.length)
        return title[randomIndex]
    }

    createTitleElement() {
        return $el("table", [
            $el("tr", [
                $el("td", [
                    $el("div.jov-title", [
                        this.headerTitle = $el("div.jov-title-header"),
                        $el("label", {
                            id: "jov-apply-button",
                            textContent: "FORCE NODES TO SYNCHRONIZE WITH PANEL? ",
                            style: {fontsize: "0.5em"}
                        }, [
                            $el("input", {
                                type: "checkbox",
                                checked: util_config.CONFIG_USER.color.overwrite,
                                style: { color: "white" },
                                onclick: (cb) => {
                                    util_config.CONFIG_USER.color.overwrite = cb.target.checked
                                    var data = {
                                        id: util_config.USER + '.color.overwrite',
                                        v: util_config.CONFIG_USER.color.overwrite
                                    }
                                    api_post('/jovimetrix/config', data)
                                    if (util_config.CONFIG_USER.color.overwrite) {
                                        node_color_all()
                                    }
                                }
                            })
                        ]),
                    ]),
                ]),
            ])
        ])
    }

    createContent() {
        const content = $el("div.comfy-modal-content", [
            this.createTitleElement(),
            $el("div.jov-config-color", [...this.createColorPalettes()]),
            $el("button", {
                id: "jov-close-button",
                type: "button",
                textContent: "CLOSE",
                onclick: () => {
                    this.close()
                    this.visible = false
                }
            })
        ])

        content.style.width = '100%'
        content.style.height = '100%'
        content.addEventListener('mousedown', this.startDrag)
        return content
    }

    constructor() {
        super()
        this.headerTitle = null
        this.overwrite = false
        this.visible = false
        this.element = $el("div.comfy-modal", { id:'jov-manager-dialog', parent: document.body }, [ this.createContent() ])
    }

	show() {
        this.visible = !this.visible
        this.headerTitle.innerText = this.createTitle()
        this.element.style.display = this.visible ? "block" : ""
    }
}
