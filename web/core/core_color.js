/**
 * File: core_color.js
 * Project: Jovimetrix
 *
 */

import { app } from '../../../scripts/app.js'
import { $el } from '../../../scripts/ui.js'
import { apiPost } from '../util/util_api.js'
import { colorContrast } from '../util/util.js'
import * as util_config from '../util/util_config.js'
import { colorPicker } from '../extern/jsColorPicker.js'

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

function nodeColorAll() {
    app.graph._nodes.forEach((node) => {
        nodeColorReset(node);
    })
    app.canvas.setDirty(true);
}

class JovimetrixPanelColorize {
    constructor() {

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        if (util_config.CONFIG_USER.color.overwrite) {
            nodeColorAll();
        }
        this.watchForColorInputs();
    }

    watchForColorInputs() {
        const observer = new MutationObserver(() => {
            if (document.querySelectorAll('.jov-panel-color-input').length > 0) {
                this.initializeColorPicker();
                observer.disconnect();
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    addDragToElement(element, dragHandle = element) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;

        const dragMouseDown = (e) => {
            e.preventDefault();
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            document.onmousemove = elementDrag;
        };

        const elementDrag = (e) => {
            e.preventDefault();
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            element.style.top = (element.offsetTop - pos2) + "px";
            element.style.left = (element.offsetLeft - pos1) + "px";
        };

        const closeDragElement = () => {
            document.onmouseup = null;
            document.onmousemove = null;
        };

        dragHandle.onmousedown = dragMouseDown;
    }

    initializeColorPicker(retries = 3) {
        console.log('Initializing color picker');
        const inputs = document.querySelectorAll('.jov-panel-color-input');
        console.log('Elements found:', inputs.length);
        if (typeof window.jsColorPicker === 'function' && inputs.length > 0) {
            // Container for the color picker
            const container = document.createElement('div');
            container.id = 'jov-panel-color-picker-container';
            container.style.position = 'absolute';
            container.style.zIndex = '10000';
            document.body.appendChild(container);

            window.jsColorPicker('.jov-panel-color-input', {
                readOnly: false,
                size: 3,
                multipleInstances: false,
                appendTo: container,
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
        } else if (retries > 0) {
            console.warn(`colorPicker not available, retrying... (${retries} attempts left)`);
            setTimeout(() => this.initializeColorPicker(retries - 1), 1000);
        } else {
            console.error('colorPicker function is not available after multiple attempts');
        }
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
                    style: { background: data.background },
                    textContent: " REGEX FILTER ",
                }, [
                    $el("input", {
                        name: `regex.${data.idx}`,
                        value: data.name,
                        onchange: function() {
                            this.updateRegexColor(data.idx, "regex", this.value);
                        }
                    }),
                ]);
            } else {
                return $el(isHeader ? "td.jov-panel-color-header" : "td", {
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
                createColorInput('title', 'T'),
                createColorInput('body', 'B'),
                createNameCell()
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
            if (a[1].category == "" || a[1].category.startsWith("_")) {
                return 1;
            } else if (b[1].category == "" || b[1].category.startsWith("_")) {
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
                }
                if (Object.prototype.hasOwnProperty.call(util_config.CONFIG_THEME, meow)) {
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

                if (Object.prototype.hasOwnProperty.call(util_config.CONFIG_THEME, cat)) {
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

app.registerExtension({
    name: "jovimetrix.color",
    async setup(app) {
        // Option for user to contrast text for better readability
        const original_color = LiteGraph.NODE_TEXT_COLOR;

        util_config.setting_make('color 🎨.contrast', 'Auto-Contrast Text', 'boolean', 'Auto-contrast the title text for all nodes for better readability', true);

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

        if (util_config.CONFIG_USER.color.overwrite) {
            nodeColorAll();
        }
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

function initializeColorPicker() {
    const elements = document.querySelectorAll('input.jov-color');
    if (elements.length) {
        colorPicker(elements, {
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
                let AHEX = this.patch.attributes.color;
                if (!AHEX) return;

                let name = this.patch.attributes.name.value;
                const parts = name.split('.');
                const part = parts.pop();
                name = parts[0];
                let api_packet = {};

                if (parts.length > 1) {
                    const idx = parts[1];
                    let data = util_config.CONFIG_REGEX[idx];
                    data[part] = AHEX.value;
                    util_config.CONFIG_REGEX[idx] = data;

                    api_packet = {
                        id: `${util_config.USER}.color.regex`,
                        v: util_config.CONFIG_REGEX
                    };
                } else {
                    const themeConfig = util_config.CONFIG_THEME[name] || (util_config.CONFIG_THEME[name] = {});
                    themeConfig[part] = AHEX.value;

                    api_packet = {
                        id: `${util_config.USER}.color.theme.${name}`,
                        v: themeConfig
                    };
                }
                apiPost("/jovimetrix/config", api_packet);
                if (util_config.CONFIG_COLOR.overwrite) {
                    nodeColorAll();
                }
            }
        });
    }
}

window.addEventListener('DOMContentLoaded', initializeColorPicker);