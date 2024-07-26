/**
 * File: core_colorize.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { $el } from "../../../scripts/ui.js"
import { apiPost } from '../util/util_api.js'
import { colorContrast } from '../util/util.js'
import * as util_config from '../util/util_config.js'

import { JovimetrixConfigDialog } from "./core_config.js"
import "../extern/jsColorPicker.js"


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

export function nodeColorAll() {
    app.graph._nodes.forEach((node) => {
        nodeColorReset(node);
    })
    app.canvas.setDirty(true);
}

app.registerExtension({
    name: "jovimetrix.colorize",
    async setup(app) {
        const original_color = LiteGraph.NODE_TEXT_COLOR;

        util_config.setting_make('color.contrast', '🇯 🎨 Auto-Contrast Text', 'boolean', 'Auto-contrast the title text for all nodes for better readability', true);

        const showButton = $el("button.comfy-settings-btn", {
            textContent: "🎨",
            style: {
                right: "82%",
                cursor: "pointer",
                display: "unset",
            },
        })

        this.config_dialog = new JovimetrixConfigDialog()

        showButton.onclick = () => {
            this.config_dialog.show()
        }

        const firstKid = document.querySelector(".comfy-settings-btn")
        const parent = firstKid.parentElement
        parent.insertBefore(showButton, firstKid.nextSibling)

        let showMenuButton;
		if (!app.menu?.element.style.display && app.menu?.settingsGroup) {
			showMenuButton = new (await import("../../../scripts/ui/components/button.js")).ComfyButton({
				icon: "palette-outline",
				action: () => showButton.click(),
				tooltip: "Jovimetrix Colorizer",
				content: "Jovimetrix Colorizer",
			});
			app.menu.settingsGroup.append(showMenuButton);
		}

        // Option for user to contrast text for better readability
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

        jsColorPicker('input.jov-color', {
            readOnly: true,
            size: 2,
            multipleInstances: false,
            appendTo: this.config_dialog.element,
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
