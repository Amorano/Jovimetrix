/**
 * File: colorize.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js"
import { $el } from "/scripts/ui.js"
import * as util from './util.js'
import { JovimetrixConfigDialog } from "./config.js"
import "../extern/jsColorPicker.js"

const ext = {
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

        function setting_make(id, pretty, tip, key, junk) {
            const local = localStorage["Comfy.Settings.jov." + id]
            const val = local ? local : util.CONFIG_USER.color[key] ? util.CONFIG_USER.color[key] : junk
            app.ui.settings.addSetting({
                id: 'jov.' + id,
                name: pretty,
                type: 'text',
                tooltip: tip,
                defaultValue: val,
                onChange(v) {
                    var data = { id: id, v: v }
                    util.api_post('/jovimetrix/config', data)
                    util.CONFIG_USER.color[key] = v
                },
            })
        }

        setting_make(util.USER + '.color.titleA', 'Group Title A 🎨🇯', 'Alternative title color for separating groups in the color configuration panel.', 'titleA', '#302929')

        setting_make(util.USER + '.color.backA', 'Group Back A 🎨🇯', 'Alternative color for separating groups in the color configuration panel.', 'backA', '#050303')

        setting_make(util.USER + '.color.titleB', 'Group Title B 🎨🇯', 'Alternative title color for separating groups in the color configuration panel.', 'titleB', '#293029')

        setting_make(util.USER + '.color.backB', 'Group Back B 🎨🇯', 'Alternative color for separating groups in the color configuration panel.', 'backB', '#030503')

        jsColorPicker('input.jov-color', {
            readOnly: true,
            size: 2,
            multipleInstances: false,
            appendTo: ext.config_dialog.element,
            noAlpha: false,
            init: function(elm, rgb) {
                elm.style.backgroundColor = elm.color || "#353535FF"
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
                    data = util.CONFIG_REGEX[idx];
                    console.info(part, data, AHEX.value)
                    data[part] = AHEX.value
                    util.CONFIG_REGEX[idx] = data
                    api_packet = {
                        id: util.USER + '.color.regex',
                        v: util.CONFIG_REGEX
                    }
                } else {
                    if (util.CONFIG_THEME[name] === undefined) {
                        util.CONFIG_THEME[name] = {}
                    }
                    util.CONFIG_THEME[name][part] = AHEX.value
                    api_packet = {
                        id: util.USER + '.color.theme.' + name,
                        v: util.CONFIG_THEME[name]
                    }

                }
                util.api_post("/jovimetrix/config", api_packet)
                if (util.CONFIG_COLOR.overwrite) {
                    util.node_color_all()
                }
            }
        })

        if (util.CONFIG_USER.color.overwrite) {
            // console.info("COLORIZED")
            util.node_color_all()
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
            let colors = util.node_color_get(nodeData);
            //this['color'] = colors?.title || "#353535";
            if (colors?.jov_set_color) {
                delete colors.jov_set_color
                this['jov_set_color'] = 1;
            }
            //this['color'] = colors?.color || "#353535";
            if (colors?.jov_set_bgcolor) {
                delete colors.jov_set_bgcolor
                this['jov_set_bgcolor'] = 1;
            }
            if (result) {
                result.serialize_widgets = true
            }
            return result
        }
    }
}

app.registerExtension(ext)

