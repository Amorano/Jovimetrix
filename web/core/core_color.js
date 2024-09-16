/**
 * File: core_color.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { apiGet, apiJovimetrix, setting_make } from "../util/util_api.js";
import { colorContrast, colorRGB2Hex } from "../util/util.js";

let PANEL_STATIC, PANEL_COLORIZE, CONTENT, NODE_LIST, CONFIG_CORE, CONFIG_USER, CONFIG_COLOR, CONFIG_REGEX, CONFIG_THEME;
const USER = "user.default";

class ColorPicker {
    constructor(config) {
        this.colorPickers = window.jsColorPicker?.colorPickers || [];
        this.docCookies = {
            getItem: (key, def) => local_get(key, def),
            setItem: (key, value) => local_set(key, value)
        };

        // Initialize with the passed config
        this.config = config;
        this.color = this.config.color || "#FFFFFF";
    }

    static createInstance(elm, config) {
        const initConfig = {
            input: elm,
            patch: elm,
            isIE8: !!document.all && !document.addEventListener,
            margin: { left: -1, top: 2 },
            customBG: "#FFFFFF",
            color: ColorPicker.extractValue(elm),
            initStyle: "display: none",
            mode: ColorPicker.docCookies["colorPickerMode"] || "hsv-h",
            memoryColors: ColorPicker.docCookies["colorPickerMemos" + ((config || {}).noAlpha ? "NoAlpha" : "")],
            size: ColorPicker.docCookies["colorPickerSize"] || 1,
            renderCallback: ColorPicker.renderCallback,
            actionCallback: ColorPicker.actionCallback
        };

        for (const n in config) {
            initConfig[n] = config[n];
        }

        return new ColorPicker(initConfig);
    }


    static extractValue(elm) {
        const val = elm.getAttribute("color") || elm.style.backgroundColor || "#353535FF";
        return val.includes("NAN") ? "#353535FF" : val;
    }

    static renderCallback(colors, mode) {
        const rgb = Object.values(colors.RND.rgb).reverse();
        const AHEX = !colors.HEX.includes("NAN") ? colorRGB2Hex(rgb) : "#353535FF";

        this.patch.style.cssText =
            "color:" + (colors.rgbaMixCustom.luminance > 0.22 ? "#222" : "#ddd") + ";" +
            "background-color: " + AHEX + ";";

        this.input.setAttribute("color", AHEX);

        if (this.displayCallback) {
            this.displayCallback(colors, mode, this);
        }
    }

    static actionCallback(event, action) {
        if (action == "toMemory") {
            const memos = this.nodes.memos;
            const cookieTXT = [];

            for (let n = 0, m = memos.length; n < m; n++) {
                let backgroundColor = memos[n].style.backgroundColor;
                let opacity = memos[n].style.opacity;
                opacity = Math.round((opacity == "" ? 1 : opacity) * 100) / 100;
                cookieTXT.push(backgroundColor.replace(/, /g, ",").replace("rgb(", "rgba(").replace(")", "," + opacity + ")"));
            }

            ColorPicker.docCookies["colorPickerMemos" + (this.noAlpha ? "NoAlpha" : ""), "'" + cookieTXT.join("","") + "'"];
        } else if (action == "resizeApp") {
            ColorPicker.docCookies["colorPickerSize", this.color.options.currentSize];
        } else if (action == "modeChange") {
            const mode = this.color.options.mode;
            ColorPicker.docCookies["colorPickerMode", mode.type + "-" + mode.z];
        }
    }

    static doEventListeners(elm, multiple, off, elms) {
        const onOff = off ? "removeEventListener" : "addEventListener";
        const focusListener = () => {
            const position = ColorPicker.getOrigin(this);
            const index = multiple ? Array.prototype.indexOf.call(elms, this) : 0;
            const colorPicker = ColorPicker.colorPickers[index] || (ColorPicker.colorPickers[index] = ColorPicker.createInstance(this, config));
            const options = colorPicker.color.options;
            const colorPickerUI = colorPicker.nodes.colorPicker;
            const appendTo = (options.appendTo || document.body);
            const isStatic = /static/.test(window.getComputedStyle(appendTo).position);
            const atrect = isStatic ? { left: 0, top: 0 } : appendTo.getBoundingClientRect();

            options.color = ColorPicker.extractValue(elm);
            colorPickerUI.style.cssText =
                "position: absolute;" + (!ColorPicker.colorPickers[index].cssIsReady ? "display: none;" : "") +
                "left:" + (position.left + options.margin.left - atrect.left) + "px;" +
                "top:" + (position.top + +this.offsetHeight + options.margin.top - atrect.top) + "px;";

            if (!multiple) {
                options.input = elm;
                options.patch = elm;
                colorPicker.setColor(ColorPicker.extractValue(elm), undefined, undefined, true);
                colorPicker.saveAsBackground();
            }

            ColorPicker.colorPickers.current = ColorPicker.colorPickers[index];
            appendTo.appendChild(colorPickerUI);

            let waitTimer = setInterval(function() {
                if (ColorPicker.colorPickers.current.cssIsReady) {
                    waitTimer = clearInterval(waitTimer);
                    colorPickerUI.style.display = "block";
                }
            }, 10);
        };

        elm[onOff]("focus", focusListener);

        if (!ColorPicker.colorPickers.evt || off) {
            ColorPicker.colorPickers.evt = true;
            window[onOff]("mousedown", (e) => {
                const colorPicker = ColorPicker.colorPickers.current;
                const colorPickerUI = (colorPicker ? colorPicker.nodes.colorPicker : undefined);
                const isColorPicker = colorPicker && (function(elm) {
                    while (elm) {
                        if ((elm.className || "").indexOf("cp-app") !== -1) return elm;
                        elm = elm.parentNode;
                    }
                    return false;
                })(e.target);

                if (isColorPicker && Array.prototype.indexOf.call(ColorPicker.colorPickers, isColorPicker)) {
                    if (e.target == colorPicker.nodes.exit) {
                        colorPickerUI.style.display = "none";
                        document.activeElement.blur();
                    }
                } else if (Array.prototype.indexOf.call(elms, e.target) !== -1) {
                    console.info("why");
                } else if (colorPickerUI) {
                    colorPickerUI.style.display = "none";
                }
            });
        }
    }
};

function colorPicker(elms, config, callback) {
    const testColors = new window.Colors({ customBG: config.customBG, allMixDetails: true });

    for (let n = 0, m = elms?.length; n < m; n++) {
        const elm = elms[n];

        if (config == "destroy") {
            ColorPicker.doEventListeners(elm, (config && config.multipleInstances), true, elms);
            if (ColorPicker.colorPickers[n]) {
                ColorPicker.colorPickers[n].destroyAll();
            }
        } else {
            const color = ColorPicker.extractValue(elm);
            const value = color.split("(");

            testColors.setColor(color);
            if (config && config.init) {
                config.init(elm, testColors.colors);
            }
            elm.setAttribute("data-colorMode", value[1] ? value[0].substr(0, 3) : "HEX");
            ColorPicker.doEventListeners(elm, (config && config.multipleInstances), false, elms);
            if (config && config.readOnly) {
                elm.readOnly = true;
            }
        }
    }

    if (callback) {
        callback(ColorPicker.colorPickers);
    }

    return ColorPicker.colorPickers;
}

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

    initializeColorPicker(container) {
        const elements = container.querySelector(".jov-panel-color-input");
        colorPicker(elements, {
            readOnly: false,
            size: 3,
            multipleInstances: false,
            appendTo: container,
            noAlpha: false,
            init: function(elm, rgb) {
                elm.style.backgroundColor = elm.color || LiteGraph.WIDGET_BGCOLOR;
                elm.style.color = rgb.RGBLuminance > 0.22 ? "#222" : "#ddd";
            },
            convertCallback: function() {
                const AHEX = this.patch.attributes.color;
                if (!AHEX) return;

                const parts = this.patch.attributes.name.value.split(".");
                const part = parts.pop();
                const name = parts.join(".");

                let key, value;

                if (parts.length > 1) {
                    CONFIG_REGEX[parts[1]][part] = AHEX.value;
                    key = `${USER}.color.regex`
                    value = CONFIG_REGEX;
                } else {
                    const themeConfig = CONFIG_THEME[name] || (CONFIG_THEME[name] = {});
                    themeConfig[part] = AHEX.value;
                    key`${USER}.color.theme.${name}`
                    value = CONFIG_THEME[name];
                }

                apiJovimetrix(key, value, "config");
                if (CONFIG_COLOR.overwrite) {
                    nodeColorAll();
                }
            }
        });
    }

    updateRegexColor = (index, key, value) => {
        CONFIG_REGEX[index][key] = value;
        apiJovimetrix(USER + ".color.regex", CONFIG_REGEX, "config");
        nodeColorAll()
    };

    templateColorRow = (data, type = "block") => {
        const isRegex = type == "regex";
        const isHeader = type == "header";
        const self = this;

        const createNameCell = () => {
            if (isRegex) {
                return $el("td", {
                    style: { backgroundColor: data.background },
                    textContent: " REGEX FILTER ",
                }, [
                    $el("input", {
                        name: `regex.${data.idx}`,
                        value: data.name,
                        onchange: function() {
                            self.updateRegexColor(data.idx, "regex", this.value);
                        }
                    }),
                ]);
            } else {
                return $el(isHeader ? "td.jov-panel-color-header" : "td", {
                    style: isHeader ? { backgroundColor: data.background } : {},
                    textContent: data.name
                });
            }
        };

        const createColorInput = (suffix, value) => {
            return $el("td", [
                $el("input.jov-panel-color-input2", {
                    value: value,
                    name: isRegex ? `regex.${data.idx}.${suffix}` : `${data.name}.${suffix}`,
                    backgroundColor: data[suffix]
                })
            ]);
        };

        return [
            $el("tr", {
                style: !isHeader ? { backgroundColor: data.background } : {}
            }, [
                createColorInput("title", "T"),
                createColorInput("body", "B"),
                createNameCell()
            ])
        ];
    };

    createRegexPalettes() {
        let colorTable = null
        const header =
            $el("table.flexible-table", [
                colorTable = $el("thead", [
                ]),
            ])

        // rule-sets first
        var idx = 0
        const rules = CONFIG_COLOR?.regex || []
        rules.forEach(entry => {
            const data = {
                idx: idx,
                name: entry.regex,
                title: entry.title, // || LiteGraph.NODE_TITLE_COLOR,
                body: entry.body, // || LiteGraph.NODE_DEFAULT_COLOR,
            }
            const row = this.templateColorRow(data, "regex");
            colorTable.appendChild($el("tbody", row))
            idx += 1
        })
        return [header];
    }

    createColorPalettes() {
        if (PANEL_STATIC !== undefined) {
            return PANEL_STATIC;
        }

        var data = {};
        let colorTable = null;
        const header =
            $el("table.flexible-table", [
                colorTable = $el("thead", [
                ]),
            ])

        // get categories to generate on the fly
        const category = []
        const all_nodes = Object.entries(NODE_LIST ? NODE_LIST : []);
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
        const alts = CONFIG_COLOR
        const background = [alts?.backA, alts?.backB]
        const background_title = [alts?.titleA, alts?.titleB]
        let background_index = 0
        all_nodes.forEach(entry => {
            var name = entry[0]
            var cat = entry[1].category
            var meow = cat.split("/")[0]

            if (!category.includes(meow))
            {
                // major category first?
                background_index = (background_index + 1) % 2
                data = {
                    name: meow,
                }
                if (Object.prototype.hasOwnProperty.call(CONFIG_THEME, meow)) {
                    data.title = CONFIG_THEME?.[meow].title
                    data.body = CONFIG_THEME?.[meow].body
                }
                colorTable.appendChild($el("tbody", this.templateColorRow(data, "header")))
                category.push(meow)
            }

            if(category.includes(cat) == false) {
                background_index = (background_index + 1) % 2
                data = {
                    name: cat,
                    background: background_title[background_index] || LiteGraph.WIDGET_BGCOLOR
                }

                if (Object.prototype.hasOwnProperty.call(CONFIG_THEME, cat)) {
                    data.title = CONFIG_THEME?.[cat].title
                    data.body = CONFIG_THEME?.[cat].body
                }
                colorTable.appendChild($el("tbody", this.templateColorRow(data, "header")))
                category.push(cat)
            }

            const who = CONFIG_THEME[name] || {};
            data = {
                name: name,
                title: who.title,
                body: who.body,
                background: background[background_index] || LiteGraph.NODE_DEFAULT_COLOR
            }
            colorTable.appendChild($el("tbody", this.templateColorRow(data, "block")))
        })
        PANEL_STATIC = [header];
        return PANEL_STATIC;
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
                    $el("div.jov-config-color", this.createRegexPalettes()),
                    $el("div.jov-config-color", this.createColorPalettes()),
                ]);

        this.initializeColorPicker(content);
        return content;
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
    /*
    async beforeRegisterNodeDef(nodeType) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this, arguments);
            if (this) {
                nodeColorReset(this, false);
            }
            return me;
        }
    },*/
    async afterConfigureGraph() {
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

            console.info("Jovimetrix Colorizer Configuration loaded");
        } catch (error) {
            console.error("Error initializing Jovimetrix Colorizer Panel:", error);
        }

        PANEL_COLORIZE = new JovimetrixPanelColorize();
        // refresh once seems to fix missing info on first use
        CONTENT = PANEL_COLORIZE.createContent();
        if (CONFIG_USER.color.overwrite) {
            nodeColorAll();
        }
        console.info("Jovimetrix Colorizer Panel initialized successfully");
    }
    /*
    async refreshComboInNodes() {

    }*/
})

document.addEventListener("keydown", (event) => {
    if (event.key === "T" || event.key === "B") {
        const elm = document.querySelector(".jov-panel-color-input");
        ColorPicker.createInstance(elm, {});  // Pass your config if needed
    }
});