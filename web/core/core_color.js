/**
 * File: core_color.js
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { apiGet } from "../util/util_api.js";
import { colorContrast } from "../util/util.js";

let PANEL_COLORIZE, NODE_LIST;

function getColor(node) {
    // regex overrides first
    const CONFIG_REGEX = app.extensionManager.setting.get('jovi.color.regex');

    for (const { regex, ...colors } of CONFIG_REGEX || []) {
        if (regex && node.type.match(new RegExp(regex, "i"))) {
            return colors;
        }
    }

    // explicit color set first...
    const CONFIG_THEME = app.extensionManager.setting.get('jovi.color.theme');
    const newColor = CONFIG_THEME?.[node.type]
        ?? (function() {
            let color = NODE_LIST?.[node.type];
            if (color?.category) {
                let k = color.category;
                while (k) {
                    if (CONFIG_THEME?.[k]) {
                        return CONFIG_THEME[k];
                    }
                    k = k.substring(0, k.lastIndexOf("/"));
                }
            }
            return null;
        })();

    return newColor;
}

const origDrawNode = LGraphCanvas.prototype.drawNode;
LGraphCanvas.prototype.drawNode = function (node, ctx) {
    // STASH THE CURRENT COLOR STATE
    const origTitle = node.constructor.title_text_color;
    const origSelectedTitleColor = LiteGraph.NODE_SELECTED_TITLE_COLOR;
    const origNodeTextColor = LiteGraph.NODE_TEXT_COLOR;
    const origWidgetSecondaryTextColor = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
    const origWidgetTextColor = LiteGraph.WIDGET_TEXT_COLOR;
    const origNodeTitleColor = LiteGraph.NODE_TITLE_COLOR;
    const origWidgetBGColor = LiteGraph.WIDGET_BGCOLOR;

    const new_color = getColor(node);

    if (new_color) {
        // Title text
        // node.constructor.title_text_color = '#00FF00'

        // Title text when node is selected
        //LiteGraph.NODE_SELECTED_TITLE_COLOR = '#FF00FF'

        //const contrast = localStorage["JOVIMETRIX 🔺🟩🔵.Color 🎨.Auto-Contrast"] || false;
        //if (contrast == true) {
            //var color = this.color || LiteGraph.NODE_TITLE_COLOR;
            ///var bgcolor = this.bgcolor || LiteGraph.WIDGET_BGCOLOR;
        if (new_color?.title) {
            node.constructor.title_text_color = colorContrast(new_color.title) ? "#000" : "#FFF";
            LiteGraph.NODE_SELECTED_TITLE_COLOR = colorContrast(new_color.title) ? "#000" : "#FFF";
        }
            //LiteGraph.NODE_TEXT_COLOR = colorContrast(bgcolor) ? "#000" : "#FFF";
        //}

        // Slot label text
        //LiteGraph.NODE_TEXT_COLOR = '#7777FF'

        // Widget Text
        //LiteGraph.WIDGET_SECONDARY_TEXT_COLOR = "#FFFFFF"

        // Widget controls + field text
        //LiteGraph.WIDGET_TEXT_COLOR = '#FF0000';

        // Widget control BG color
        // LiteGraph.WIDGET_BGCOLOR

        // node's title bar background color
        if (new_color?.title) {
            // LiteGraph.NODE_TITLE_COLOR = new_color.title;
            //node.constructor.color = new_color.title;
            node.color = new_color.title;
        }

        // node's body background color
        if (new_color?.body) {
            // LiteGraph.WIDGET_BGCOLOR = new_color.body;
            node.bgcolor = new_color.body;
        }
    }

    const res = origDrawNode.apply(this, arguments);

    // Default back to last pushed state ComfyUI colors
    if (new_color) {
        node.constructor.title_text_color = origTitle;
        LiteGraph.NODE_SELECTED_TITLE_COLOR = origSelectedTitleColor;
        LiteGraph.NODE_TEXT_COLOR = origNodeTextColor;
        LiteGraph.WIDGET_SECONDARY_TEXT_COLOR = origWidgetSecondaryTextColor;
        LiteGraph.WIDGET_TEXT_COLOR = origWidgetTextColor;
        LiteGraph.NODE_TITLE_COLOR = origNodeTitleColor;
        LiteGraph.WIDGET_BGCOLOR = origWidgetBGColor;
    }

    return res;
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
                    onclick: async () => {
                        await this.applyColor();
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

    async hidePicker(cancelled = false) {
        if (this.picker) {
            try {
                this.pickerWrapper.style.display = 'none';
                if (!cancelled && this.currentButton) {
                    const newColor = this.picker.color.hexString;
                    this.currentButton.style.backgroundColor = newColor;
                    this.currentButton.dataset.color = newColor;
                    await this.updateConfig(newColor);
                }
                this.currentButton = null;
            } catch (error) {
                console.error('Error hiding picker:', error);
            }
        }
    }

    async applyColor() {
        if (this.currentButton && this.picker) {
            const newColor = this.picker.color.hexString;
            this.currentButton.style.backgroundColor = newColor;
            this.currentButton.dataset.color = newColor;
            await this.updateConfig(newColor);
            await this.hidePicker();
        }
    }

    async updateConfig(newColor) {
        const [type, index, colorType] = this.currentButton.dataset.identifier.split('.');
        console.info(type, index, colorType);
        if (type === '') {
            const CONFIG_REGEX = app.extensionManager.setting.get('jovi.color.regex');
            CONFIG_REGEX[index][colorType] = newColor;
            try {
                await app.extensionManager.setting.set("jovi.color.regex", CONFIG_REGEX);
            } catch (error) {
                console.error(`Error changing setting: ${error}`);
            }
        } else {
            const CONFIG_THEME = app.extensionManager.setting.get('jovi.color.theme');
            CONFIG_THEME[type] = CONFIG_THEME[type] || (CONFIG_THEME[type] = {});
            CONFIG_THEME[type][colorType] = newColor;
            try {
                await app.extensionManager.setting.set("jovi.color.theme", CONFIG_THEME);
            } catch (error) {
                console.error(`Error changing setting: ${error}`);
            }
            //apiJovimetrix(`${USER}.color.theme.${type}`, CONFIG_THEME[type], "config");
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

        const CONFIG_REGEX = app.extensionManager.setting.get('jovi.color.regex') || [];
        CONFIG_REGEX.forEach((entry, idx) => {
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

        let background_index = 0;
        const categories = [];
        const CONFIG_THEME = app.extensionManager.setting.get('jovi.color.theme');

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
                    body: CONFIG_THEME?.[category]?.body
                };
                this.tbody.appendChild(this.templateColorRow(element, null, "jov-panel-color-cat_minor"));
                categories.push(category);
            }

            const nodeConfig = CONFIG_THEME[name] || {};
            const data = {
                name: name,
                title: nodeConfig.title,
                body: nodeConfig.body
            };
            this.tbody.appendChild(this.templateColorRow(data));
        });
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
    settings: [
        {
            id: "jovi.color.regex",
            name: "Regex Entries for Jovimetrix Colorizer",
            type: "hidden",
            defaultValue: {}
        },
        {
            id: "jovi.color.theme",
            name: "Node theme entries for Jovimetrix Colorizer",
            type: "hidden",
            defaultValue: {}
        },
    ],
    async afterConfigureGraph() {

        [NODE_LIST] = await Promise.all([
            apiGet("/object_info")
        ]);

        if (!app.extensionManager.setting.get('jovi.color.regex')) {
            let CONFIG_CORE
            try {
                [CONFIG_CORE] = await Promise.all([
                    apiGet("/jovimetrix/config")
                ]);

                const CONFIG_REGEX = CONFIG_CORE.user.default.color.regex || [];
                const CONFIG_THEME = CONFIG_CORE.user.default.color.theme;

                try {
                    await app.extensionManager.setting.set("jovi.color.regex", CONFIG_REGEX);
                } catch (error) {
                    console.error(`Error changing setting: ${error}`);
                }

                try {
                    await app.extensionManager.setting.set("jovi.color.theme", CONFIG_THEME);
                } catch (error) {
                    console.error(`Error changing setting: ${error}`);
                }

                console.info("Jovimetrix Colorizer Panel initialized successfully");
            } catch (error) {
                console.error("Error initializing Jovimetrix Colorizer Panel:", error);
            }
        }
    }
});