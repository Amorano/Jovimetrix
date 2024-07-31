/**
 * File: core_color.js
 * Project: Jovimetrix
 *
 */

import { app } from '../../../scripts/app.js'
import { ComfyDialog, $el } from '../../../scripts/ui.js'
import { apiPost } from '../util/util_api.js'
import { colorContrast } from '../util/util.js'
import * as util_config from '../util/util_config.js'
import '../extern/jsColorPicker.js'

// gets the CONFIG entry for name
function nodeColorGet(node) {
    const find_me = node.type || node.name;
    if (find_me === undefined) {
        return
    }

    // First look to regex....
    for (const colors of util_config.CONFIG_REGEX) {
        if (colors.regex == "") {
            continue
        }
        const regex = new RegExp(colors.regex, 'i');
        const found = find_me.match(regex);
        if (found !== null && found[0].length > 0) {
            return colors;
        }
    }

    // now look to theme
    let color = util_config.CONFIG_THEME[find_me]
    if (color) {
        return color
    }

    color = util_config.NODE_LIST[find_me]
    // now look to category theme
    if (color && color.category) {
        const segments = color.category.split('/')
        let k = segments.join('/')
        while (k) {
            const found = util_config.CONFIG_THEME[k]
            if (found) {
                return found
            }
            const last = k.lastIndexOf('/')
            k = last !== -1 ? k.substring(0, last) : ''
        }
    }
    return null;
}

// refresh the color of a node
function nodeColorReset(node, refresh=true) {
    const color = nodeColorGet(node);
    if (color) {
        if (color.body) {
            node.bgcolor = color.body;
        }
        if (color.title) {
            node.color = color.title;
        }
        if (refresh) {
            node?.graph?.setDirtyCanvas(true, true);
        }
    }
}

function nodeColorList(nodes) {
    Object.entries(nodes).forEach((node) => {
        nodeColorReset(node, false);
    })
    app.canvas.setDirty(true);
}

function nodeColorAll() {
    app.graph._nodes.forEach((node) => {
        nodeColorReset(node);
    })
    app.canvas.setDirty(true);
}

/*
const colorClear = (name) => {
    apiPost("/jovimetrix/config/clear", { name });
    delete util_config.CONFIG_THEME[name];
    if (util_config.CONFIG_COLOR.overwrite) nodeColorAll();
};
*/

class JovimetrixConfigDialog extends ComfyDialog {
    constructor() {
        super();
        this.headerTitle = null;
        this.visible = false;
        this.element = $el("div.comfy-modal", { id: 'jov-manager-dialog', parent: document.body }, [this.createContent()]);
        this.element.addEventListener('mousedown', this.startDrag);
    }

    createColorInput = (value, name, color) => $el("input.jov-color", { value, name, color });

    templateColorBlock = (data) => [
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

    templateColorHeader = (data) => [
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

    templateColorRegex = ({ idx, name, background, title, body }) => (
        $el("tr", [
            $el("td", { style: { background } }, [
                $el("input", {
                    name: `regex.${idx}`,
                    value: name,
                    onchange: (e) => this.updateRegexColor(idx, "regex", e.target.value)
                })
            ]),
            $el("td", [this.createColorInput("T", `regex.${idx}.title`, title)]),
            $el("td", [this.createColorInput("B", `regex.${idx}.body`, body)])
        ])
    );

    updateRegexColor = (index, key, value) => {
        util_config.CONFIG_REGEX[index][key] = value;
        apiPost("/jovimetrix/config", {
            id: `${util_config.USER}.color.regex`,
            v: util_config.CONFIG_REGEX
        });
        nodeColorAll();
    };

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
        const colorTable = $el("table");

        util_config.CONFIG_COLOR.regex?.forEach((entry, idx) => {
            colorTable.appendChild($el("tbody", this.templateColorRegex({
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
                colorTable.appendChild($el("tbody", this.templateColorHeader({
                    name: meow,
                    background: LiteGraph.WIDGET_BGCOLOR,
                    ...util_config.CONFIG_THEME[meow]
                })));
                category.push(meow);
            }

            if (!category.includes(cat)) {
                backgroundIndex = (backgroundIndex + 1) % 2;
                colorTable.appendChild($el("tbody", this.templateColorHeader({
                    name: cat,
                    background: backgroundTitles[backgroundIndex] || LiteGraph.WIDGET_BGCOLOR,
                    ...util_config.CONFIG_THEME[cat]
                })));
                category.push(cat);
            }

            colorTable.appendChild($el("tbody", this.templateColorBlock({
                name,
                ...util_config.CONFIG_THEME[name],
                background: backgrounds[backgroundIndex] || LiteGraph.NODE_DEFAULT_COLOR
            })));
        });

        return [colorTable];
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
        return $el("div.jov-panel-color", {
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
const DIALOG_COLOR = new JovimetrixConfigDialog();

app.registerExtension({
    name: "jovimetrix.color",
    async setup(app) {

        // Option for user to contrast text for better readability
        const original_color = LiteGraph.NODE_TEXT_COLOR;

        util_config.setting_make('color.contrast', '🇯 🎨 Auto-Contrast Text', 'boolean', 'Auto-contrast the title text for all nodes for better readability', true);

        const drawNodeShape = LGraphCanvas.prototype.drawNodeShape;
        LGraphCanvas.prototype.drawNodeShape = function() {
            const contrast = localStorage["Comfy.Settings.jov.user.default.color.contrast"] || false;
            if (contrast == true) {
                var color = this.color || LiteGraph.NODE_TITLE_COLOR;
                var bgcolor = this.bgcolor || LiteGraph.WIDGET_BGCOLOR;
                this.node_title_color = colorContrast(color) ? "#000" : "#FFF";
                LiteGraph.NODE_TEXT_COLOR = colorContrast(bgcolor) ? "#000" : "#FFF";
            } else {
                this.node_title_color = original_color
                LiteGraph.NODE_TEXT_COLOR = original_color;
            }
            drawNodeShape.apply(this, arguments);
        };

        const showButton = $el("button.comfy-settings-btn", {
            textContent: "🎨",
            style: {
                right: "82%",
                cursor: "pointer",
                display: "unset",
            },
        });

        // Old Style Popup Node Color Changer
        if (app.menu?.element.style.display || !app.menu?.settingsGroup) {
            showButton.onclick = () => {
                DIALOG_COLOR.show()
            }

            const firstKid = document.querySelector(".comfy-settings-btn")
            const parent = firstKid.parentElement;
            parent.insertBefore(showButton, firstKid.nextSibling);
        }
        // New Style Panel Node Color Changer
        else if (!app.menu?.element.style.display && app.menu?.settingsGroup && !app.extensionManager) {
            const showMenuButton = new (await import("../../../scripts/ui/components/button.js")).ComfyButton({
                icon: "palette-outline",
                action: () => showButton.click(),
                tooltip: "Jovimetrix Colorizer",
                content: "Jovimetrix Colorizer",
            });
            app.menu.settingsGroup.append(showMenuButton);
        }

        if (util_config.CONFIG_USER.color.overwrite) {
            nodeColorAll();
        }

        document.addEventListener('DOMContentLoaded', function() {
            jsColorPicker('input.jov-color', {
                readOnly: true,
                size: 2,
                multipleInstances: false,
                appendTo: document,
                noAlpha: false,
                init: function(elm, rgb) {
                    elm.style.backgroundColor = elm.color || LiteGraph.WIDGET_BGCOLOR;
                    elm.style.color = rgb.RGBLuminance > 0.22 ? '#222' : '#ddd'
                },
                convertCallback: function(data) {
                    var AHEX = this.patch.attributes.color
                    if (AHEX === undefined) return
                    var name = this.patch.attributes.name.value
                    const parts = name.split('.')
                    const part = parts.slice(-1)[0]
                    name = parts[0]
                    let api_packet = {}
                    if (parts.length > 2) {
                        const idx = parts[1];
                        data = util_config.CONFIG_REGEX[idx];
                        data[part] = AHEX.value
                        util_config.CONFIG_REGEX[idx] = data
                        api_packet = {
                            id: util_config.USER + '.color.regex',
                            v: util_config.CONFIG_REGEX
                        }
                    } else {
                        if (util_config.CONFIG_THEME[name] === undefined) {
                            util_config.CONFIG_THEME[name] = {}
                        }
                        util_config.CONFIG_THEME[name][part] = AHEX.value
                        api_packet = {
                            id: util_config.USER + '.color.theme.' + name,
                            v: util_config.CONFIG_THEME[name]
                        }
                    }
                    apiPost("/jovimetrix/config", api_packet);
                    if (util_config.CONFIG_COLOR.overwrite) {
                        nodeColorAll();
                    }
                }
            });
        });
    },
    async beforeRegisterNodeDef(nodeType) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this, arguments);
            if (this) {
                nodeColorReset(this, false);
            }
            return me;
        }
    }
})

const new_menu = app.ui.settings.getSettingValue("Comfy.UseNewMenu", "Disabled");

class JovimetrixPanelColorize {
    constructor() {

        if (util_config.CONFIG_USER.color.overwrite) {
            nodeColorAll();
        }

        document.addEventListener('DOMContentLoaded', function() {
            jsColorPicker('.jov-panel-color-input', {
                readOnly: true,
                size: 2,
                multipleInstances: false,
                appendTo: document,
                noAlpha: false,
                init: function(elm, rgb) {
                    elm.style.backgroundColor = elm.color || LiteGraph.WIDGET_BGCOLOR;
                    elm.style.color = rgb.RGBLuminance > 0.22 ? '#222' : '#ddd'
                },
                convertCallback: function(data) {
                    var AHEX = this.patch.attributes.color
                    if (AHEX === undefined) return
                    var name = this.patch.attributes.name.value
                    const parts = name.split('.')
                    const part = parts.slice(-1)[0]
                    name = parts[0]
                    let api_packet = {}
                    if (parts.length > 2) {
                        const idx = parts[1];
                        data = util_config.CONFIG_REGEX[idx];
                        data[part] = AHEX.value
                        util_config.CONFIG_REGEX[idx] = data
                        api_packet = {
                            id: util_config.USER + '.color.regex',
                            v: util_config.CONFIG_REGEX
                        }
                    } else {
                        if (util_config.CONFIG_THEME[name] === undefined) {
                            util_config.CONFIG_THEME[name] = {}
                        }
                        util_config.CONFIG_THEME[name][part] = AHEX.value
                        api_packet = {
                            id: util_config.USER + '.color.theme.' + name,
                            v: util_config.CONFIG_THEME[name]
                        }
                    }
                    apiPost("/jovimetrix/config", api_packet);
                    if (util_config.CONFIG_COLOR.overwrite) {
                        nodeColorAll();
                    }
                }
            })
        })
    }

    updateRegexColor = (index, key, value) => {
        util_config.CONFIG_REGEX[index][key] = value
        let api_packet = {
            id: util_config.USER + '.color.regex',
            v: util_config.CONFIG_REGEX
        }
        apiPost("/jovimetrix/config", api_packet)
        nodeColorAll()
    };

    templateColorRow = (data, type = 'block') => {
        const isRegex = type === 'regex';
        const isHeader = type === 'header';

        const createNameCell = () => {
            if (isRegex) {
                return $el("td", {
                    style: { background: data.background }
                }, [
                    $el("input", {
                        name: `regex.${data.idx}`,
                        value: data.name,
                        onchange: function() {
                            this.updateRegexColor(data.idx, "regex", this.value);
                        }
                    })
                ]);
            } else {
                return $el(isHeader ? "td.jov-config-color-header" : "td", {
                    style: isHeader ? { background: data.background } : {},
                    textContent: data.name
                });
            }
        };

        const createColorInput = (suffix, value) => {
            return $el("td", [
                $el("input.jov-panel-color-input", {
                    value: value,
                    name: isRegex ? `regex.${data.idx}.${suffix}` : `${data.name}.${suffix}`,
                    color: data[suffix]
                })
            ]);
        };

        return [
            $el("tr", {
                style: !isHeader ? { background: data.background } : {}
            }, [
                createNameCell(),
                createColorInput('title', 'T'),
                createColorInput('body', 'B')
            ])
        ];
    };

    createColorPalettes() {
        var data = {}
        let colorTable = null
        const header =
            $el("table.flexible-table", [
                colorTable = $el("thead", [
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
            const row = this.templateColorRow(data, 'regex');
            colorTable.appendChild($el("tbody", row))
            idx += 1
        })

        // get categories to generate on the fly
        const category = []
        const all_nodes = Object.entries(util_config?.NODE_LIST ? util_config.NODE_LIST : []);
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
                colorTable.appendChild($el("tbody", this.templateColorRow(data, 'header')))
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
                colorTable.appendChild($el("tbody", this.templateColorRow(data, 'header')))
                category.push(cat)
            }

            const who = util_config.CONFIG_THEME[name] || {};
            data = {
                name: name,
                title: who.title,
                body: who.body,
                background: background[background_index] || LiteGraph.NODE_DEFAULT_COLOR
            }
            colorTable.appendChild($el("tbody", this.templateColorRow(data, 'block')))
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
        return $el("div.jov-title", [
            this.headerTitle = $el("div.jov-title-header"),
            $el("label", {
                id: "jov-apply-button",
                textContent: "FORCE NODES TO SYNCHRONIZE WITH PANEL? ",
                style: {fontsize: "0.5em"}
            }, [
                $el("input", {
                    type: "checkbox",
                    checked: util_config.CONFIG_USER.color.overwrite,
                    onclick: (cb) => {
                        util_config.CONFIG_USER.color.overwrite = cb.target.checked
                        var data = {
                            id: util_config.USER + '.color.overwrite',
                            v: util_config.CONFIG_USER.color.overwrite
                        }
                        apiPost('/jovimetrix/config', data)
                        if (util_config.CONFIG_USER.color.overwrite) {
                            nodeColorAll()
                        }
                    }
                })
            ]),
        ])
    }

    createContent() {
        const content = $el("div.jov-panel-color", [
            this.createTitleElement(),
            $el("div.jov-config-color", [...this.createColorPalettes()]),
        ])
        content.addEventListener('mousedown', this.startDrag)
        return content
    }
}

let PANEL_COLORIZE;

if(new_menu != "Disabled" && app.extensionManager) {
    app.extensionManager.registerSidebarTab({
        id: "jovimetrix.sidebar.colorizer",
        icon: "pi pi-palette",
        title: "JOVIMETRIX COLORIZER 🔺🟩🔵",
        tooltip: "Colorize your nodes how you want; I'm not your dad.",
        type: "custom",
        render: async (el) => {
            if (PANEL_COLORIZE === undefined) {
                PANEL_COLORIZE = new JovimetrixPanelColorize();
                const content = PANEL_COLORIZE.createContent();
                el.appendChild(content);
            }
        }
    });
}