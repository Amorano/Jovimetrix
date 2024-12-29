/**
 * File: core_color.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { apiGet, apiJovimetrix, setting_make } from "../util/util_api.js";
import { colorContrast } from "../util/util.js";

const USER = "user.default";

let PANEL_COLORIZE, NODE_LIST, CONFIG_CORE, CONFIG_USER, CONFIG_COLOR, CONFIG_REGEX, CONFIG_THEME;

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

function applyTheme(theme) {
    const majorElements = document.querySelectorAll('.jov-panel-color-cat_major');
    const minorElements = document.querySelectorAll('.jov-panel-color-cat_minor');
    majorElements.forEach(el => el.classList.remove('light', 'dark'));
    minorElements.forEach(el => el.classList.remove('light', 'dark'));
    majorElements.forEach(el => el.classList.add(theme));
    minorElements.forEach(el => el.classList.add(theme));
}

class JovimetrixPanelColorize {
    constructor() {
        this.content = null;
        this.currentButton = null;
        this.picker = null;
        this.pickerWrapper = null;
        this.recentColors = [];
        this.title_content = "HI!"
        this.searchInput = null;
        this.tbody = null;
    }

    createSearchInput() {
        this.searchInput = $el("input", {
            type: "text",
            placeholder: "Filter nodes...",
            className: "jov-search-input",
            oninput: (e) => this.filterItems(e.target.value)
        });
        return this.searchInput;
    }

    filterItems(searchTerm) {
        if (!this.tbody) return;

        const searchLower = searchTerm.toLowerCase();
        const rows = this.tbody.querySelectorAll('tr');

        rows.forEach(row => {
            const nameCell = row.querySelector('td:last-child');
            if (!nameCell) return;

            const text = nameCell.textContent.toLowerCase();
            const categoryMatch = row.classList.contains('jov-panel-color-cat_major') ||
                                row.classList.contains('jov-panel-color-cat_minor');

            // Show categories if they or their children match
            if (categoryMatch) {
                const siblingRows = this.getNextSiblingRowsUntilCategory(row);
                const hasVisibleChildren = siblingRows.some(sibling => {
                    const siblingText = sibling.querySelector('td:last-child')?.textContent.toLowerCase() || '';
                    return siblingText.includes(searchLower);
                });

                row.style.display = hasVisibleChildren || text.includes(searchLower) ? '' : 'none';
            } else {
                row.style.display = text.includes(searchLower) ? '' : 'none';
            }
        });
    }

    getNextSiblingRowsUntilCategory(categoryRow) {
        const siblings = [];
        let currentRow = categoryRow.nextElementSibling;

        while (currentRow &&
               !currentRow.classList.contains('jov-panel-color-cat_major') &&
               !currentRow.classList.contains('jov-panel-color-cat_minor')) {
            siblings.push(currentRow);
            currentRow = currentRow.nextElementSibling;
        }

        return siblings;
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

        button.addEventListener('mousedown', (event) => {
            event.stopPropagation();
            this.currentButton = button;
            this.showPicker(event.target, button.dataset.color);
        });

        return button;
    }

    showPicker(buttonElement, color) {
        if (!this.picker) {
            try {
                this.pickerWrapper = $el('div.picker-wrapper', {
                    style: {
                        position: 'absolute',
                        zIndex: '9999',
                        backgroundColor: '#fff',
                        padding: '5px',
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

                const resetButton = $el('button', {
                    textContent: 'Reset',
                    onclick: () => {
                        this.picker.color.set(LiteGraph.NODE_DEFAULT_COLOR);
                    }
                });

                const applyButton = $el('button', {
                    textContent: 'Apply',
                    onclick: () => {
                        this.applyColor();
                    }
                });

                buttonWrapper.appendChild(cancelButton);
                buttonWrapper.appendChild(resetButton);
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
                            component: iro.ui.Slider,
                            options: { sliderType: 'hue' }
                        },
                        {
                            component: iro.ui.Slider,
                            options: { sliderType: 'value' }
                        },
                        {
                            component: iro.ui.Slider,
                            options: { sliderType: 'saturation' }
                        },
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
        if (this.picker) {
            try {
                this.picker.color.set(color || '#ffffff');
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
                this.updateRecentColors(color);
            } catch (error) {
                console.error('Error showing picker:', error);
            }
        } else {
            console.error('Picker not created successfully');
        }
    }

    hidePicker(cancelled = false) {
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

    templateColorRow2(data, type, classList="jov-panel-color-category") {
        const titleColor = data.title || LiteGraph.NODE_DEFAULT_COLOR;
        const bodyColor = data.body || LiteGraph.NODE_DEFAULT_COLOR;

        const element = $el("tr", {}, [
            $el("td", {}, [this.createColorButton("T", titleColor, `${data.name}.${data.idx}.title`)]),
            $el("td", {}, [this.createColorButton("B", bodyColor, `${data.name}.${data.idx}.body`)]),
            (type === "regex") ? $el("td", [
                $el("input", {
                    value: data.name
                })
              ])
            : $el("td", { textContent: data.name })
        ]);
        if (classList) {
            element.classList.add(classList);
        }
        return element;
    }

    templateColorRow(data, type, classList = "jov-panel-color-category") {
        const titleColor = data.title || LiteGraph.NODE_DEFAULT_COLOR;
        const bodyColor = data.body || LiteGraph.NODE_DEFAULT_COLOR;

        // Determine background color based on class
        let rowClass = classList;
        let style = {};

        if (classList === "jov-panel-color-cat_major") {
             // Darker background for major categories
            style.backgroundColor = "var(--border-color)";
        } else if (classList === "jov-panel-color-cat_minor") {
            style.backgroundColor = "var(--tr-odd-bg-color)";
        }

        const element = $el("tr", { className: rowClass, style }, [
            $el("td", {}, [this.createColorButton("T", titleColor, `${data.name}.${data.idx}.title`)]),
            $el("td", {}, [this.createColorButton("B", bodyColor, `${data.name}.${data.idx}.body`)]),
            (type === "regex") ? $el("td", [
                $el("input", {
                    value: data.name
                })
            ])
            : $el("td", { textContent: data.name })
        ]);

        return element;
    }

    createRegexPalettes() {
        const table = $el("table.flexible-table");
        this.tbody = $el("tbody");

        (CONFIG_COLOR?.regex || []).forEach((entry, idx) => {
            const data = {
                idx: idx,
                name: entry.regex,
                title: entry.title || LiteGraph.NODE_TITLE_COLOR,
                body: entry.body || LiteGraph.NODE_DEFAULT_COLOR,
            };
            this.tbody.appendChild(this.templateColorRow(data, "regex"));
        });

        table.appendChild(this.tbody);
        return table;
    }

    createColorPalettes() {
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
                const element = {
                    name: majorCategory,
                    title: CONFIG_THEME?.[majorCategory]?.title,
                    body: CONFIG_THEME?.[majorCategory]?.body,
                };
                this.tbody.appendChild(this.templateColorRow(element, null, "jov-panel-color-cat_major"));
                categories.push(majorCategory);
            }

            if (!categories.includes(category)) {
                background_index = (background_index + 1) % 2;
                const element = {
                    name: category,
                    title: CONFIG_THEME?.[category]?.title,
                    body: CONFIG_THEME?.[category]?.body,
                    background: background_title[background_index] || LiteGraph.WIDGET_BGCOLOR
                };
                this.tbody.appendChild(this.templateColorRow(element, null, "jov-panel-color-cat_minor"));
                categories.push(category);
            }

            const nodeConfig = CONFIG_THEME[name] || {};
            const data = {
                name: name,
                title: nodeConfig.title,
                body: nodeConfig.body,
                background: background[background_index] || LiteGraph.NODE_DEFAULT_COLOR
            };
            this.tbody.appendChild(this.templateColorRow(data));
        });

        //table.appendChild(tbody);
        //return table;
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
            const table = this.createRegexPalettes();
            this.createColorPalettes();

            this.title_content = $el("div.jov-title-header", { textContent: "EMPTY" });
            this.content = $el("div.jov-panel-color", [
                $el("div.jov-title", [this.title_content]),
                this.createSearchInput(),  // Add search input
                $el("div.jov-config-color", [table]),
                $el("div.button", []),
            ]);

            // hide the picker when clicking outside
            document.addEventListener('click', (event) => {
                if (this.picker && !this.pickerWrapper.contains(event.target) && !event.target.classList.contains('color-button')) {
                    this.hidePicker(true);
                }
            });
        }
        applyTheme('light');
        this.title_content.textContent = this.getRandomTitle();
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