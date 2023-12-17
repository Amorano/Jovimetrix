/**
 * File: colorize.js
 * Project: Jovimetrix
 *
 */

import { app } from "/scripts/app.js";
import { $el } from "/scripts/ui.js";
import * as util from './util.js';
import { JovimetrixConfigDialog } from "./config.js";
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
        });

        this.config_dialog = new JovimetrixConfigDialog();

        showButton.onclick = () => {
            this.config_dialog.show();
        };

        const firstKid = document.querySelector(".comfy-settings-btn")
        const parent = firstKid.parentElement;
        parent.insertBefore(showButton, firstKid.nextSibling);
    },
    async setup(app) {

        function setting_make(id, pretty, tip, key, junk) {
            const local = localStorage["Comfy.Settings.jov." + id];
            const val = local ? local : util.CONFIG_USER.color[key] ? util.CONFIG_USER.color[key] : junk;
            app.ui.settings.addSetting({
                id: 'jov.' + id,
                name: pretty,
                type: 'text',
                tooltip: tip,
                defaultValue: val,
                onChange(v) {
                    var data = { id: id, v: v }
                    util.api_post('/jovimetrix/config', data);
                    util.CONFIG_USER.color[key] = v;
                },
            });
        }

        setting_make(util.USER + '.color.titleA', 'Group Title A 🎨🇯', 'Alternative title color for separating groups in the color configuration panel.', 'titleA', '#000');

        setting_make(util.USER + '.color.backA', 'Group Back A 🎨🇯', 'Alternative color for separating groups in the color configuration panel.', 'backA', '#000');

        setting_make(util.USER + '.color.titleB', 'Group Title B 🎨🇯', 'Alternative title color for separating groups in the color configuration panel.', 'titleB', '#000');

        setting_make(util.USER + '.color.backB', 'Group Back A 🎨🇯', 'Alternative color for separating groups in the color configuration panel.', 'backB', '#000');

        jsColorPicker('input.jov-color', {
            readOnly: true,
            size: 2,
            multipleInstances: false,
            appendTo: ext.config_dialog.element,
            noAlpha: false,
            init: function(elm, rgb) {
                elm.style.backgroundColor = elm.color || "#4D4D4D";
                elm.style.color = rgb.RGBLuminance > 0.22 ? '#222' : '#ddd';
            },
            convertCallback: function(data, options) {
                var AHEX = this.patch.attributes.color;
                if (AHEX === undefined) return;
                AHEX = AHEX.value;
                var name = this.patch.attributes.name.value;
                const parts = name.split('.');
                const part = parts.slice(-1)[0]
                name = parts[0]
                let api_packet = {};

                if (parts.length > 2) {
                    const idx = parts[1];
                    var regex = util.CONFIG_COLOR.regex || [];
                    if (idx > regex.length) {
                        var data = {
                            "regex": name,
                            [part]: AHEX,
                        }
                        util.CONFIG_COLOR.regex.push(data);
                    } else {
                        var data = regex[idx] || {};
                        data["regex"] = name;
                        data[part] = AHEX;
                        util.CONFIG_COLOR.regex[idx] = data;
                    }

                    api_packet = {
                        id: util.USER + '.color.regex',
                        v: util.CONFIG_COLOR.regex
                    }
                } else {
                    if (util.THEME[name] === undefined) {
                        util.THEME[name] = {};
                    }
                    util.THEME[name][part] = AHEX;
                    api_packet = {
                        id: util.USER + '.color.theme.' + name,
                        v: { [part]: AHEX }
                    }
                    // console.info(api_packet)
                }
                util.api_post("/jovimetrix/config", api_packet);
                if (util.CONFIG_COLOR.overwrite) {
                    util.node_color_all();
                }
            }
        });

        if (util.CONFIG_USER.color.overwrite) {
            util.node_color_all();
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        let node = util.node_color_get(nodeData.name);
        if (node === undefined) {
            var data = {};
            if (nodeData.color) {
                data['title'] = nodeData.color
            }
            if (nodeData.bgcolor === undefined) {
                data['body'] = nodeData.bgcolor
            }
            if (data.length > 0) {
                util.THEME[nodeData.name]
                node = util.node_color_get(nodeData.name);
            }
        }

        if (node) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                // console.info(nodeData.name, node);
                if (nodeData.color === undefined) {
                    this['color'] = (node.title || '#4D4D4D')
                }

                if (nodeData.bgcolor === undefined) {
                    this['bgcolor'] = (node.body || '#4D4D4D')
                }
                /*
                // default, box, round, card
                if (nodeData.shape === undefined || nodeData.shape == false) {
                    this['_shape'] = nodeData._shape ? nodeData._shape : found.shape ?
                    found.shape in ['default', 'box', 'round', 'card'] : 'round';
                }*/
                // console.info('jovi-colorized', this.title, this.color, this.bgcolor, this._shape);
                // result.serialize_widgets = true
                return result;
            }
        }
    }
}

app.registerExtension(ext);

