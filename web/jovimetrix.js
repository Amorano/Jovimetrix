/**
 * File: jovimetrix.js
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js";
import { ComfyDialog, $el } from "../../../scripts/ui.js";
import { template_color_block } from './template.js'
import * as util from './util.js';
import './extern/color.all.min.js'

var headID = document.getElementsByTagName("head")[0];
var cssNode = document.createElement('link');
cssNode.rel = 'stylesheet';
cssNode.type = 'text/css';
cssNode.href = 'extensions/Jovimetrix/jovimetrix.css';
headID.appendChild(cssNode);

export function renderTemplate(template, data) {
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
            template = template.replace(regex, data[key]);
        }
    }
    return template;
}

let config_overwrite = false;
const CONFIG = await util.CONFIG();
const NODE_LIST = await util.NODE_LIST();

export let jovimetrix = null;

class JovimetrixConfigDialog extends ComfyDialog {
    createElements() {
        let colorTable = null;
        const header =
            $el("div.jov-config-column", [
                $el("table", [
                    colorTable = $el("thead", [
                    ]),
                ]),
            ]);

        var existing = [];
        const COLORS = Object.entries(CONFIG.color);
        COLORS.forEach(entry => {
            existing.push(entry[0]);
            var data = {
                name: entry[0],
                title: entry[1].title,
                body: entry[1].body
            };
            const html = renderTemplate(template_color_block, data);
            colorTable.innerHTML += html;
        });

        // now the rest which are untracked and their "categories"
        var categories = [];
        const nodes = Object.entries(NODE_LIST);
        nodes.forEach(entry => {
            var name = entry[0];
            if (existing.includes(name) == false) {
                var data = {
                    name: entry[0],
                    title: '#7F7F7FEE',
                    body: '#7F7F7FEE',
                };
                const html = renderTemplate(template_color_block, data);
                colorTable.innerHTML += html;
            }

            var cat = entry[1].category;
            if (categories.includes(cat) == false) {
                categories.push(cat);
            }
        });

        categories.sort(function (a, b) {
            return a.toLowerCase().localeCompare(b.toLowerCase());
        });

        Object.entries(categories).forEach(entry => {
            if (existing.includes(entry[1]) == false) {
                var data = {
                    name: entry[1],
                    title: '#3F3F3FEE',
                    body: '#3F3F3FEE',
                };
                const html = renderTemplate(template_color_block, data);
                colorTable.innerHTML += html;
            }
        });
		return [header];
	}

    constructor() {
        super();

        const init = async () => {
            const content =
                $el("div.comfy-modal-content", [
                    $el("tr.jov-title", [
                            $el("font", {size:6, color:"white"}, [`JOVIMETRIX COLOR CONFIGURATION`])]
                        ),
                    $el("div.jov-menu-container", [...this.createElements()]),
                    $el("button", {
                        id: "jov-close-button",
                        type: "button",
                        textContent: "CLOSE",
                        onclick: () => this.close()
                    }),
                    $el("label", {
                            id: "jov-apply-button"
                        }, [
                            $el("input", {
                                type: "checkbox",
                                checked: config_overwrite,
                                onclick: (cb) => {
                                    config_overwrite = cb.target.checked;
                                }
                            })
                        ])
                    ]);


            content.style.width = '100%';
            content.style.height = '100%';
            this.element = $el("div.comfy-modal", { id:'jov-manager-dialog', parent: document.body }, [ content ]);
        };
        init();
	}

	show() {
		this.element.style.display = "block";
	}
}

class Jovimetrix {
    // gets the CONFIG entry for this Node.type || Node.name
    node_color_get(find_me) {
        let node = CONFIG.color[find_me];
        if (node) {
            return node;
        }
        node = NODE_LIST[find_me];
        //console.info(find_me, node);

        if (node && node.category) {
            const segments = node.category.split('/');
            let k = segments.join('/');
            while (k) {
                const found = CONFIG.color[k];
                if (found) {
                    // console.log(found, node.category);
                    return found;
                }
                const last = k.lastIndexOf('/');
                k = last !== -1 ? k.substring(0, last) : '';
            }
        }
    }

    // refresh the color of a node
    node_color_reset(node, refresh=true) {
        const data = this.node_color_get(node.type || node.name);
        if (data) {
            node.bgcolor = data.body;
            node.color = data.title;
            // console.info(node, data);
            if (refresh) {
                node.setDirtyCanvas(true, true);
            }
        }
    }

    node_color_list(nodes) {
        Object.entries(nodes).forEach((node) => {
            this.node_color_reset(node, false);
        });
        app.graph.setDirtyCanvas(true, true);
    }

    node_color_all() {
        app.graph._nodes.forEach((node) => {
            this.node_color_reset(node, false);
        });
        app.graph.setDirtyCanvas(true, true);
    }

    constructor() {
        this.settings = new JovimetrixConfigDialog();
    }
}
jovimetrix = new Jovimetrix();

export function color_clear(name) {
    var body = {
        "name": name,
    }
    util.api_post("/jovimetrix/config/clear", body);
    delete CONFIG.color[name];
}

(function (global) {
	if (typeof global.ColorPicker === 'undefined') {
        global.ColorPicker = {};
    }

	// Define docCookies if it's not already defined
	if (typeof window.ColorPicker.docCookies === 'undefined') {
		window.ColorPicker.docCookies = {
			getItem: function (key, def) {
				const data = util.local_get(key, def);
                return data;
			},
			setItem: function (key, value, options) {
                util.local_set(key, value);
			}
		};
	}

	global.jsColorPicker = function(selectors, config) {
		var renderCallback = function(colors, mode) {
            // console.info(colors);
            var options = this,
                input = options.input,
                patch = options.patch,
                RGB = colors.RND.rgb;

            // console.info(colors);
            const AHEX = util.convert_hex(colors);
            patch.style.cssText =
                'color:' + (colors.rgbaMixCustom.luminance > 0.22 ? '#222' : '#ddd') + ';' + // Black...???
                'background-color: ' + AHEX + ';' +
                'filter:';

            input.setAttribute("color", AHEX);
            if (options.displayCallback) {
                options.displayCallback(colors, mode, options);
            }
        },
        extractValue = function(elm) {
            const val = elm.getAttribute('color') || elm.style.backgroundColor || '#7F7F7FEE';
            if (val.includes("NAN")) {
                return "#7F7F7FEE";
            }
            return val;
        },
        actionCallback = function(event, action) {
            var options = this,
                colorPicker = colorPickers.current;

            if (action === 'toMemory') {
                var memos = colorPicker.nodes.memos,
                    backgroundColor = '',
                    opacity = 0,
                    cookieTXT = [];

                for (var n = 0, m = memos.length; n < m; n++) {
                    backgroundColor = memos[n].style.backgroundColor;
                    opacity = memos[n].style.opacity;
                    opacity = Math.round((opacity === '' ? 1 : opacity) * 100) / 100;
                    cookieTXT.push(backgroundColor.
                        replace(/, /g, ',').
                        replace('rgb(', 'rgba(').
                        replace(')', ',' + opacity + ')')
                    );
                }
                cookieTXT = '\'' + cookieTXT.join('\',\'') + '\'';
                util.local_set('colorPickerMemos' + (options.noAlpha ? 'NoAlpha' : ''), cookieTXT);
            } else if (action === 'resizeApp') {
                util.local_set('colorPickerSize', colorPicker.color.options.currentSize);
            } else if (action === 'modeChange') {
                var mode = colorPicker.color.options.mode;
                util.local_set('colorPickerMode', mode.type + '-' + mode.z);
            }
        },
        createInstance = function(elm, config) {
            var initConfig = {
                    klass: global.ColorPicker,
                    input: elm,
                    patch: elm,
                    isIE8: !!document.all && !document.addEventListener, // Opera???
                    // *** animationSpeed: 200,
                    // draggable: true,
                    margin: {left: -1, top: 2},
                    customBG: '#FFFFFF',
                    // displayCallback: displayCallback,
                    /* --- regular colorPicker options from this point --- */
                    color: extractValue(elm),
                    initStyle: 'display: none',
                    mode: ColorPicker.docCookies('colorPickerMode') || 'hsv-h',

                    memoryColors: ColorPicker.docCookies('colorPickerMemos'),
                    size: ColorPicker.docCookies('colorPickerSize') || 1,
                    renderCallback: renderCallback,
                    actionCallback: actionCallback
                };

            for (var n in config) {
                initConfig[n] = config[n];
            }
            return new initConfig.klass(initConfig);
        },
        doEventListeners = function(elm, multiple, off) {
            var onOff = off ? 'removeEventListener' : 'addEventListener',
                focusListener = function(e) {
                    var input = this,
                        position = global.ColorPicker.getOrigin(input),
                        index = multiple ? Array.prototype.indexOf.call(elms, this) : 0,
                        colorPicker = colorPickers[index] ||
                            (colorPickers[index] = createInstance(this, config)),
                            options = colorPicker.color.options,
                            colorPickerUI = colorPicker.nodes.colorPicker,
                            appendTo = (options.appendTo || document.body),
                            isStatic = /static/.test(global.getComputedStyle(appendTo).position),
                            atrect = isStatic ? {left: 0, top: 0} : appendTo.getBoundingClientRect(),
                            waitTimer = 0;

                    options.color = extractValue(elm); // brings color to default on reset
                    colorPickerUI.style.cssText =
                        'position: absolute;' + (!colorPickers[index].cssIsReady ? 'display: none;' : '') +
                        'left:' + (position.left + options.margin.left - atrect.left) + 'px;' +
                        'top:' + (position.top + +input.offsetHeight + options.margin.top - atrect.top) + 'px;';

                    if (!multiple) {
                        options.input = elm;
                        options.patch = elm; // check again???
                        colorPicker.setColor(extractValue(elm), undefined, undefined, true);
                        colorPicker.saveAsBackground();
                    }
                    colorPickers.current = colorPickers[index];
                    appendTo.appendChild(colorPickerUI);
                    waitTimer = setInterval(function() { // compensating late style on onload in colorPicker
                        if (colorPickers.current.cssIsReady) {
                            waitTimer = clearInterval(waitTimer);
                            colorPickerUI.style.display = 'block';
                        }
                    }, 10);
                },
                mousDownListener = function(e) {
                    var colorPicker = colorPickers.current,
                        colorPickerUI = (colorPicker ? colorPicker.nodes.colorPicker : undefined),
                        animationSpeed = colorPicker ? colorPicker.color.options.animationSpeed : 0,
                        isColorPicker = colorPicker && (function(elm) {
                            while (elm) {
                                if ((elm.className || '').indexOf('cp-app') !== -1) return elm;
                                elm = elm.parentNode;
                            }
                            return false;
                        })(e.target),
                        inputIndex = Array.prototype.indexOf.call(elms, e.target);

                    if (isColorPicker && Array.prototype.indexOf.call(colorPickers, isColorPicker)) {
                        if (e.target === colorPicker.nodes.exit) {
                            colorPickerUI.style.display = 'none';
                            document.activeElement.blur();
                        } else {
                            // ...
                        }
                    } else if (inputIndex !== -1) {
                        // ...
                    } else if (colorPickerUI) {
                        colorPickerUI.style.display = 'none';
                    }
                };

            elm[onOff]('focus', focusListener);

            if (!colorPickers.evt || off) {
                colorPickers.evt = true; // prevent new eventListener for global

                global[onOff]('mousedown', mousDownListener);
            }
        },
        // this is a way to prevent data binding on HTMLElements
        colorPickers = global.jsColorPicker.colorPickers || [],
        elms = document.querySelectorAll(selectors),
        testColors = new global.Colors({customBG: config.customBG, allMixDetails: true});
		global.jsColorPicker.colorPickers = colorPickers;

		for (var n = 0, m = elms.length; n < m; n++) {
			var elm = elms[n];

			if (config === 'destroy') {
				doEventListeners(elm, (config && config.multipleInstances), true);
				if (colorPickers[n]) {
					colorPickers[n].destroyAll();
				}
			} else {
				var color = extractValue(elm);
				var value = color.split('(');

				testColors.setColor(color);
				if (config && config.init) {
					config.init(elm, testColors.colors);
				}
				elm.setAttribute('data-colorMode', value[1] ? value[0].substr(0, 3) : 'HEX');
				doEventListeners(elm, (config && config.multipleInstances), false);
				if (config && config.readOnly) {
					elm.readOnly = true;
				}
			}
		};

		return global.jsColorPicker.colorPickers;
    };

    global.ColorPicker.docCookies = function(key, value, def) {
        if (value === undefined) {
			const data = util.local_get(key, def);
            return data
        } else {
			util.local_set(key, value);
        }
    };
})(typeof window !== 'undefined' ? window : this);

jsColorPicker('input.jov-color', {
    readOnly: true,
    size: 2,
    multipleInstances: false,
    appendTo: jovimetrix.settings.element,
    noAlpha: false,
    init: function(elm, rgb) {
        elm.style.backgroundColor = elm.getAttribute("color");
        elm.style.color = rgb.RGBLuminance > 0.22 ? '#222' : '#ddd';
    },
    convertCallback: function(data, options) {
        const AHEX = util.convert_hex(data);
        var name = this.patch.attributes.name.value;
        var part = this.patch.attributes.part.value;
        // {title:'', body:'', shape: ''}
        let color = CONFIG.color[name];
        if (color === undefined){
            CONFIG.color[name] = {}
        }
        CONFIG.color[name][part] = AHEX;

        if (config_overwrite) {
            // console.info(name, part, CONFIG.color[name][part])
            jovimetrix.node_color_all();
        }

        // for the API
        color = {
            "name": name,
            "part": part,
            "color": AHEX
        }
        util.api_post("/jovimetrix/config", color);
    },
});
