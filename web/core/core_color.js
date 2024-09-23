/**
 * File: core_color.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { apiGet, apiJovimetrix, setting_make } from "../util/util_api.js";
import { colorContrast } from "../util/util.js";

let PANEL_COLORIZE, NODE_LIST, CONFIG_CORE, CONFIG_USER, CONFIG_COLOR, CONFIG_REGEX, CONFIG_THEME;
const USER = "user.default";

// Recursively get color by category in theme
function getColorByCategory(find_me) {
    let color = NODE_LIST?.[find_me];
    if (color?.category) {
        let k = color.category;
        while (k) {
            if (CONFIG_THEME?.[k]) return CONFIG_THEME[k];
            k = k.substring(0, k.lastIndexOf("/"));
        }
    }
}

// Get the CONFIG entry for a node by name
function nodeColorGet(node) {
    // Look for matching regex pattern
    for (const { regex, ...colors } of CONFIG_REGEX || []) {
        if (regex && node.type.match(new RegExp(regex, "i"))) return colors;
    }
    // Check theme first by node name, then by category
    return CONFIG_THEME?.[node.type] || getColorByCategory(node.type);
}

// Refresh the color of a node
function nodeColorReset(node, refresh = true) {
    const color = nodeColorGet(node);
    if (color) {
        node.bgcolor = color.body || node.bgcolor;
        node.color = color.title || node.color;
    }
    if (refresh) node?.graph?.setDirtyCanvas(true, true);
}

// Apply color to all nodes
function nodeColorAll() {
    app.graph._nodes.forEach(nodeColorReset);
    app.canvas.setDirty(true);
}

class JovimetrixPanelColorize {
    constructor() {
        this.content = null;
        this.currentButton = null;
        this.picker = null;
        this.pickerWrapper = null;
        this.recentColors = [];
    }

    createColorButton(label, color, identifier) {
        const button = $el('button.color-button', {
            style: { backgroundColor: color },
            dataset: { color: color, identifier: identifier },
            value: label,
            content: label,
            textContent: label,
            label: label
        });

        button.addEventListener('click', (event) => {
            // console.log('Button clicked:', label, color, identifier);
            event.stopPropagation();
            this.currentButton = button;
            this.showPicker(event.target, button.dataset.color);
        });

        return button;
    }

    createPicker() {
        if (!this.picker) {
            try {
                this.pickerWrapper = $el('div.picker-wrapper', {
                    style: {
                        position: 'absolute',
                        zIndex: '9999',
                        backgroundColor: '#fff',
                        padding: '10px',
                        borderRadius: '5px',
                        boxShadow: '0 0 10px rgba(0,0,0,0.2)',
                        display: 'none'
                    }
                });

                const pickerElement = $el('div.picker');
                const recentColorsElement = $el('div.recent-colors');
                const buttonWrapper = $el('div.button-wrapper', {
                    style: {
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginTop: '10px'
                    }
                });

                const cancelButton = $el('button', {
                    textContent: 'Cancel',
                    onclick: () => this.hidePicker(true)
                });

                const applyButton = $el('button', {
                    textContent: 'Apply',
                    onclick: () => this.applyColor()
                });

                buttonWrapper.appendChild(cancelButton);
                buttonWrapper.appendChild(applyButton);

                this.pickerWrapper.appendChild(pickerElement);
                this.pickerWrapper.appendChild(recentColorsElement);
                this.pickerWrapper.appendChild(buttonWrapper);

                document.body.appendChild(this.pickerWrapper);

                this.picker = new iro.ColorPicker(pickerElement, {
                    width: 200,
                    color: '#ffffff',
                    display: 'block',
                    layout: [
                        {
                            component: iro.ui.Box,
                        },
                        {
                            component: iro.ui.Slider,
                            options: { sliderType: 'hue' }
                        },
                        {
                            component: iro.ui.Slider,
                            options: { sliderType: 'alpha' }
                        }
                    ]
                });

                this.picker.on('color:change', (color) => {
                    if (this.currentButton) {
                        this.currentButton.style.backgroundColor = color.hexString;
                    }
                });
            } catch (error) {
                console.error('Error creating Picker:', error);
            }
        }
    }

    showPicker(buttonElement, color) {
        console.log('Showing picker for button:', buttonElement, 'with color:', color);
        if (!this.picker) {
            this.createPicker();
        }
        if (this.picker) {
            try {
                this.picker.color.set(color || '#ffffff');

                // Position picker
                const buttonRect = buttonElement.getBoundingClientRect();
                const pickerRect = this.pickerWrapper.getBoundingClientRect();

                let left = buttonRect.left;
                let top = buttonRect.bottom + 5;

                if (left + pickerRect.width > window.innerWidth) {
                    left = window.innerWidth - pickerRect.width - 5;
                }

                if (top + pickerRect.height > window.innerHeight) {
                    top = buttonRect.top - pickerRect.height - 5;
                }

                this.pickerWrapper.style.left = `${left}px`;
                this.pickerWrapper.style.top = `${top}px`;
                this.pickerWrapper.style.display = 'block';

                // Update recent colors
                this.updateRecentColors(color);

                // console.log('Picker shown at position:', left, top);
            } catch (error) {
                console.error('Error showing picker:', error);
            }
        } else {
            console.error('Picker not created successfully');
        }
    }

    hidePicker(cancelled = false) {
        console.log('Hiding picker');
        if (this.picker) {
            try {
                this.pickerWrapper.style.display = 'none';
                if (!cancelled && this.currentButton) {
                    const newColor = this.picker.color.hexString;
                    this.currentButton.style.backgroundColor = newColor;
                    this.currentButton.dataset.color = newColor;
                    this.updateConfig(newColor);
                }
                this.currentButton = null;
                console.log('Picker hidden');
            } catch (error) {
                console.error('Error hiding picker:', error);
            }
        }
    }

    applyColor() {
        if (this.currentButton && this.picker) {
            const newColor = this.picker.color.hexString;
            this.currentButton.style.backgroundColor = newColor;
            this.currentButton.dataset.color = newColor;
            this.updateConfig(newColor);
            this.hidePicker();
        }
    }

    updateConfig(newColor) {
        const [type, index, colorType] = this.currentButton.dataset.identifier.split('.');
        if (type === 'regex') {
            CONFIG_REGEX[index][colorType] = newColor;
            apiJovimetrix(`${USER}.color.regex`, CONFIG_REGEX, "config");
        } else {
            const themeConfig = CONFIG_THEME[type] || (CONFIG_THEME[type] = {});
            themeConfig[colorType] = newColor;
            apiJovimetrix(`${USER}.color.theme.${type}`, CONFIG_THEME[type], "config");
        }
        if (CONFIG_COLOR.overwrite) {
            nodeColorAll();
        }
    }

    updateRecentColors(color) {
        if (!this.recentColors.includes(color)) {
            this.recentColors.unshift(color);
            if (this.recentColors.length > 5) {
                this.recentColors.pop();
            }
        }

        const recentColorsElement = this.pickerWrapper.querySelector('.recent-colors');
        recentColorsElement.innerHTML = '';
        this.recentColors.forEach(recentColor => {
            const colorSwatch = $el('div', {
                style: {
                    width: '20px',
                    height: '20px',
                    backgroundColor: recentColor,
                    display: 'inline-block',
                    margin: '2px',
                    cursor: 'pointer'
                },
                onclick: () => this.picker.color.set(recentColor)
            });
            recentColorsElement.appendChild(colorSwatch);
        });
    }

    templateColorRow(data, type = "block") {
        const isRegex = type === "regex";
        const nameCell = isRegex
            ? $el("td", [
                $el("input", {
                    value: data.name,
                    onchange: (e) => this.updateColor(data.idx, e.target.value)
                })
              ])
            :
            $el("td", { textContent: data.name });

        return $el("tr", {}, [
            $el("td", {}, [this.createColorButton("T", data.title, `${data.name}.${data.idx}.title`)]),
            $el("td", {}, [this.createColorButton("B", data.body, `${data.name}.${data.idx}.body`)]),
            nameCell
        ]);
    }

    createRegexPalettes() {
        const table = $el("table.flexible-table");
        const tbody = $el("tbody");

        (CONFIG_COLOR?.regex || []).forEach((entry, idx) => {
            const data = {
                idx: idx,
                name: entry.regex,
                title: entry.title || LiteGraph.NODE_TITLE_COLOR,
                body: entry.body || LiteGraph.NODE_DEFAULT_COLOR,
            };
            tbody.appendChild(this.templateColorRow(data, "regex"));
        });

        table.appendChild(tbody);
        return table;
    }

    createColorPalettes() {
        const table = $el("table.flexible-table");
        const tbody = $el("tbody");

        const all_nodes = Object.entries(NODE_LIST || []).sort((a, b) => {
            const categoryComparison = a[1].category.toLowerCase().localeCompare(b[1].category.toLowerCase());
            return categoryComparison;
        });

        const alts = CONFIG_COLOR;
        const background = [alts?.backA, alts?.backB];
        const background_title = [alts?.titleA, alts?.titleB];
        let background_index = 0;
        const categories = [];

        all_nodes.forEach(([name, node]) => {
            const category = node.category;
            const majorCategory = category.split("/")[0];

            if (!categories.includes(majorCategory)) {
                background_index = (background_index + 1) % 2;
                const data = {
                    name: majorCategory,
                    title: CONFIG_THEME?.[majorCategory]?.title,
                    body: CONFIG_THEME?.[majorCategory]?.body,
                };
                tbody.appendChild(this.templateColorRow(data, "header"));
                categories.push(majorCategory);
            }

            if (!categories.includes(category)) {
                background_index = (background_index + 1) % 2;
                const data = {
                    name: category,
                    title: CONFIG_THEME?.[category]?.title,
                    body: CONFIG_THEME?.[category]?.body,
                    background: background_title[background_index] || LiteGraph.WIDGET_BGCOLOR
                };
                tbody.appendChild(this.templateColorRow(data, "category"));
                categories.push(category);
            }

            const nodeConfig = CONFIG_THEME[name] || {};
            const data = {
                name: name,
                title: nodeConfig.title,
                body: nodeConfig.body,
                background: background[background_index] || LiteGraph.NODE_DEFAULT_COLOR
            };
            tbody.appendChild(this.templateColorRow(data, "block"));
        });

        table.appendChild(tbody);
        return table;
    }

    getRandomTitle() {
        const TITLES = [
            "COLOR CONFIGURATION", "COLOR CALIBRATION", "COLOR CUSTOMIZATION",
            "CHROMA CALIBRATION", "CHROMA CONFIGURATION", "CHROMA CUSTOMIZATION",
            "CHROMATIC CALIBRATION", "CHROMATIC CONFIGURATION", "CHROMATIC CUSTOMIZATION",
            "HUE HOMESTEAD", "PALETTE PREFERENCES", "PALETTE PERSONALIZATION",
            "PALETTE PICKER", "PIGMENT PREFERENCES", "PIGMENT PERSONALIZATION",
            "PIGMENT PICKER", "SPECTRUM STYLING", "TINT TAILORING", "TINT TWEAKING"
        ];
        return TITLES[Math.floor(Math.random() * TITLES.length)];
    }

    createContent() {
        if (!this.content) {
            this.content = $el("div.jov-panel-color", [
                $el("div.jov-title", [
                    $el("div.jov-title-header", { textContent: this.getRandomTitle() }),
                ]),
                $el("div.jov-config-color", [this.createRegexPalettes()]),
                $el("div.jov-config-color", [this.createColorPalettes()]),
            ]);

            // Add a global click event listener to hide the picker when clicking outside
            document.addEventListener('click', (event) => {
                if (this.picker && !this.pickerWrapper.contains(event.target) && !event.target.classList.contains('color-button')) {
                    this.hidePicker(true);
                }
            });
        }
        return this.content;
    }
}

app.extensionManager.registerSidebarTab({
    id: "jovimetrix.sidebar.colorizer",
    icon: "pi pi-palette",
    title: "JOVIMETRIX COLORIZER 🔺🟩🔵",
    tooltip: "Color node title and body via unique name, group and regex filtering\nJOVIMETRIX 🔺🟩🔵",
    type: "custom",
    render: async (el) => {
        el.innerHTML = "";
        if (!PANEL_COLORIZE) {
            PANEL_COLORIZE = new JovimetrixPanelColorize();
        }
        const content = PANEL_COLORIZE.createContent();
        el.appendChild(content);
    }
});

app.registerExtension({
    name: "jovimetrix.color",
    async setup() {
        // Option for user to contrast text for better readability
        const original_color = LiteGraph.NODE_TEXT_COLOR;

        function colorAll(checked) {
            CONFIG_USER.color.overwrite = checked;
            apiJovimetrix(`${USER}.color.overwrite`, CONFIG_USER.color.overwrite, "config");
            if (CONFIG_USER?.color?.overwrite) {
                nodeColorAll();
            }
        }

        setting_make("Color 🎨", "Auto-Contrast", "boolean",
            "Auto-contrast the title text for all nodes for better readability",
            true
        );

        setting_make("Color 🎨", "Synchronize", "boolean",
            "Synchronize color updates from color panel",
            true,
            {},
            [],
            colorAll
        );

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
    },
    async beforeRegisterNodeDef(nodeType) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this, arguments);
            nodeColorReset(this, false);
            return me;
        }
    },
    async afterConfigureGraph() {
        if (!CONFIG_USER) {
            console.info("Initializing Jovimetrix Colorizer Panel");
            try {
                [NODE_LIST, CONFIG_CORE] = await Promise.all([
                    apiGet("/object_info"),
                    apiGet("/jovimetrix/config")
                ]);
                CONFIG_USER = CONFIG_CORE.user.default;
                CONFIG_COLOR = CONFIG_USER.color;
                CONFIG_REGEX = CONFIG_COLOR.regex || [];
                CONFIG_THEME = CONFIG_COLOR.theme;
            } catch (error) {
                console.error("Error initializing Jovimetrix Colorizer Panel:", error);
            }
            console.info("Jovimetrix Colorizer Panel initialized successfully");
        }

        if (CONFIG_USER.color.overwrite) {
            nodeColorAll();
        }
    }
});