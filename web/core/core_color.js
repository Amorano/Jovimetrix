/**
 * File: core_color.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { apiGet, apiJovimetrix, setting_make } from "../util/util_api.js";
import { colorContrast } from "../util/util.js";

let PICKER, PANEL_COLORIZE, CONTENT, NODE_LIST, CONFIG_CORE, CONFIG_USER, CONFIG_COLOR, CONFIG_REGEX, CONFIG_THEME;
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

    createColorButton(label, color, identifier) {
        const button = $el('button.color-button', {
            style: { backgroundColor: color },
            dataset: { color: color, identifier: identifier },
            value: label,
            content: label,
            textContent: label,
            label: label
        });

        button.addEventListener('click', () => {
            PICKER.setColor(button.dataset.color || '#ffffff');
            showColorPicker(button);
        });

        return button;
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
        const content = $el("div.jov-panel-color", [
            $el("div.jov-title", [
                $el("div.jov-title-header", { textContent: this.getRandomTitle() }),
            ]),
            $el("div.jov-config-color", [this.createRegexPalettes()]),
            $el("div.jov-config-color", [this.createColorPalettes()]),
        ]);

        // this.initializeColorPicker(content);
        return content;
    }
}

// Function to create or show the color picker on a button press
function showColorPicker(button) {
    const colorValue = button.dataset.color || '#ffffff';
    if (!PICKER) {
        // Initialize the color picker once
        PICKER = Pickr.create({
            el: button,
            theme: 'classic', // or any theme
            default: colorValue, // Set the correct initial color
            components: {
                // Main components
                preview: true,
                opacity: true,
                hue: true,

                // Input / interaction
                interaction: {
                    hex: true,
                    rgba: true,
                    input: true,
                    save: true
                }
            }
        });

        PICKER.on('save', (color) => {
            const newColor = color.toHEXA().toString();
            button.style.backgroundColor = newColor;
            button.dataset.color = newColor;

            const [type, index, colorType] = button.dataset.identifier.split('.');

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

            PICKER.hide();
        });
    }
    PICKER.setColor(colorValue);
    PICKER.show();
}

app.extensionManager.registerSidebarTab({
    id: "jovimetrix.sidebar.colorizer",
    icon: "pi pi-palette",
    title: "JOVIMETRIX COLORIZER 🔺🟩🔵",
    tooltip: "Color node title and body via unique name, group and regex filtering\nJOVIMETRIX 🔺🟩🔵",
    type: "custom",
    render: async (el) => {
        el.innerHTML = "";
        CONTENT = PANEL_COLORIZE.createContent();
        el.appendChild(CONTENT);
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

        if (CONTENT === undefined) {
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

            PANEL_COLORIZE = new JovimetrixPanelColorize();
            // refresh once seems to fix missing info on first use
            CONTENT = PANEL_COLORIZE.createContent();
            console.info("Jovimetrix Colorizer Panel initialized successfully");
        }

        if (CONFIG_USER.color.overwrite) {
            nodeColorAll();
        }
    }
})
