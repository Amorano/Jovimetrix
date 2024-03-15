/**
 * File: core_colorize.js
 * Project: Jovimetrix
 *
 */

import { app } from "../../../scripts/app.js"
import { $el } from "../../../scripts/ui.js"
import { api_post } from '../util/util_api.js'
import { color_contrast, node_color_all, node_color_get} from '../util/util_color.js'
import * as util_config from '../util/util_config.js'
import { JovimetrixConfigDialog } from "./core_config.js"
import "../extern/jsColorPicker.js"

app.registerExtension({
    name: "jovimetrix.colorize",
    async init(app) {
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
    },
    async setup(app) {

        const original_color = LiteGraph.NODE_TEXT_COLOR;

        function setting_make(id, pretty, type, tip, key, value,) {
            const _id = 'jov.' + id;
            const local = localStorage["Comfy.Settings.jov." + id]
            value = local ? local : util_config.CONFIG_USER.color[key] ? util_config.CONFIG_USER.color[key] : value;
            util_config.setting_make(_id, pretty, type, tip, value, (val) => {
                var data = { id: id, v: val }
                api_post('/jovimetrix/config', data);
                util_config.CONFIG_USER.color[key] = val;
            });
        }

        setting_make(util_config.USER + '.color.tooltips', '🇯 🎨 Tooltip Color ', 'text', 'Color to display tooltip text on ctrl-shift', 'tooltips', '#72FF27')

        setting_make(util_config.USER + '.color.titleA', '🇯 🎨 Group Title A ', 'text', 'Alternative title color for separating groups in the color configuration panel', 'titleA', '#302929')

        setting_make(util_config.USER + '.color.backA', '🇯 🎨 Group Back A ', 'text', 'Alternative color for separating groups in the color configuration panel', 'backA', '#050303');

        setting_make(util_config.USER + '.color.titleB', '🇯 🎨 Group Title B', 'text', 'Alternative title color for separating groups in the color configuration panel', 'titleB', '#293029');

        setting_make(util_config.USER + '.color.backB', '🇯 🎨 Group Back B', 'text', 'Alternative color for separating groups in the color configuration panel', 'backB', '#030503');

        setting_make(util_config.USER + '.color.contrast', '🇯 🎨 Auto-Contrast Text', 'boolean', 'Auto-contrast the title text for all nodes for better readability', 'contrast', true);

        // Option for user to contrast text for better readability
        const drawNodeShape = LGraphCanvas.prototype.drawNodeShape;
        LGraphCanvas.prototype.drawNodeShape = function() {
            const contrast = localStorage["Comfy.Settings.jov." + util_config.USER + '.color.contrast'] || false;
            if (contrast) {
                var color = this.current_node.color || LiteGraph.NODE_TITLE_COLOR;
                var bgcolor = this.current_node.bgcolor || LiteGraph.WIDGET_BGCOLOR;
                this.node_title_color = color_contrast(color);
                LiteGraph.NODE_TEXT_COLOR = color_contrast(bgcolor);
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
            init: function(elm, rgb) {
                elm.style.backgroundColor = elm.color || LiteGraph.WIDGET_BGCOLOR;
                elm.style.color = rgb.RGBLuminance > 0.22 ? '#222' : '#ddd'
            },
            convertCallback: function(data, options) {
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
                    // console.info(part, data, AHEX.value)
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
                api_post("/jovimetrix/config", api_packet)
                if (util_config.CONFIG_COLOR.overwrite) {
                    node_color_all()
                }
            }
        })

        if (util_config.CONFIG_USER.color.overwrite) {
            // console.info("COLORIZED")
            node_color_all()
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this, arguments)
            let colors = node_color_get(nodeData);

            if (colors?.title) {
                this['color'] = colors.title
            }
            if (colors?.body) {
                this['bgcolor'] = colors.body
            }
            if (colors?.jov_set_color) {
                delete colors.jov_set_color
            }
            if (colors?.jov_set_bgcolor) {
                delete colors.jov_set_bgcolor
            }
            if (me) {
                me.serialize_widgets = true
            }
            return me
        }
    }
})
