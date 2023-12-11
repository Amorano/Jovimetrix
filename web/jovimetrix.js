/**
 * File: jovimetrix.js
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js";
import { JovimetrixConfigDialog } from "./config.js";
import * as util from './util.js';
import './extern/color.all.min.js'

export let jovimetrix = null;

class Jovimetrix {
    // gets the CONFIG entry for this Node.type || Node.name
    node_color_get(find_me) {
        let node = util.CONFIG.color[find_me];
        if (node) {
            return node;
        }
        node = util.NODE_LIST[find_me];
        //console.info(node);
        if (node && node.category) {
            //console.info(util.CONFIG);
            const segments = node.category.split('/');
            let k = segments.join('/');
            while (k) {
                const found = util.CONFIG.color[k];
                if (found) {
                    //console.info(found, node.category);
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

    setup() {
        jovimetrix.node_color_all();
    }

    constructor() {
        this.config = new JovimetrixConfigDialog();
    }
}
jovimetrix = new Jovimetrix();

export function color_clear(name) {
    var body = {
        "name": name,
    }
    util.api_post("/jovimetrix/config/clear", body);
    delete util.CONFIG.color[name];
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
    appendTo: jovimetrix.config.element,
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
        let color = util.CONFIG.color[name];
        if (color === undefined){
            util.CONFIG.color[name] = {}
        }
        util.CONFIG.color[name][part] = AHEX;

        if (jovimetrix.config.overwrite) {
            console.info(name, part, util.CONFIG.color[name][part])
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
