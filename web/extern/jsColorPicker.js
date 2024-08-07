// jsColorpicker.js

import { colorRGB2Hex } from '../util/util.js';
import { local_set, local_get } from '../util/util_config.js';

(function (global) {
    if (typeof global.ColorPicker === 'undefined') {
        global.ColorPicker = {};
    }

    // Define docCookies if it's not already defined
    if (typeof window.ColorPicker.docCookies === 'undefined') {
        window.ColorPicker.docCookies = {
            getItem: function (key, def) {
                const data = local_get(key, def);
                return data;
            },
            setItem: function (key, value) {
                local_set(key, value);
            }
        };
    }

    global.jsColorPicker = function(selectors, config) {
        var renderCallback = function(colors, mode) {
                var options = this,
                    input = options.input,
                    patch = options.patch,
                    RGB = colors.RND.rgb;
                const rgb = Object.values(RGB).reverse();
                const AHEX = !colors.HEX.includes("NAN") ? colorRGB2Hex(rgb) : LiteGraph.NODE_DEFAULT_BGCOLOR // LiteGraph.NODE_DEFAULT_COLOR; // LiteGraph.NODE_DEFAULT_BGCOLOR

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
                const val = elm.getAttribute('color') || elm.style.backgroundColor || '#353535FF';
                if (val.includes("NAN")) {
                    return "#353535FF";
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
                    ColorPicker.docCookies('colorPickerMemos' + (options.noAlpha ? 'NoAlpha' : ''), cookieTXT);
                } else if (action === 'resizeApp') {
                    ColorPicker.docCookies('colorPickerSize', colorPicker.color.options.currentSize);
                } else if (action === 'modeChange') {
                    var mode = colorPicker.color.options.mode;
                    ColorPicker.docCookies('colorPickerMode', mode.type + '-' + mode.z);
                }
            },
            createInstance = function(elm, config) {
                var initConfig = {
                    klass: window.ColorPicker,
                    input: elm,
                    patch: elm,
                    isIE8: !!document.all && !document.addEventListener, // Opera???
                    // *** animationSpeed: 200,
                    // *** draggable: true,
                    margin: {left: -1, top: 2},
                    customBG: '#FFFFFF',
                    // displayCallback: displayCallback,
                    /* --- regular colorPicker options from this point --- */
                    color: extractValue(elm),
                    initStyle: 'display: none',
                    mode: ColorPicker.docCookies('colorPickerMode') || 'hsv-h',
                    // memoryColors: (function(colors, config) {
                    // 	return config.noAlpha ?
                    // 		colors.replace(/\,\d*\.*\d*\)/g, ',1)') : colors;
                    // })($.docCookies('colorPickerMemos'), config || {}),
                    memoryColors: ColorPicker.docCookies('colorPickerMemos' +
                        ((config || {}).noAlpha ? 'NoAlpha' : '')),
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
                    focusListener = function() {
                        var input = this,
                            position = window.ColorPicker.getOrigin(input),
                            index = multiple ? Array.prototype.indexOf.call(elms, this) : 0,
                            colorPicker = colorPickers[index] ||
                                (colorPickers[index] = createInstance(this, config)),
                            options = colorPicker.color.options,
                            colorPickerUI = colorPicker.nodes.colorPicker,
                            appendTo = (options.appendTo || document.body),
                            isStatic = /static/.test(window.getComputedStyle(appendTo).position),
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
                        waitTimer = setInterval(function() {
                            // compensating late style on onload in colorPicker
                            if (colorPickers.current.cssIsReady) {
                                waitTimer = clearInterval(waitTimer);
                                colorPickerUI.style.display = 'block';
                            }
                        }, 10);
                    },
                    mousDownListener = function(e) {
                        var colorPicker = colorPickers.current,
                            colorPickerUI = (colorPicker ? colorPicker.nodes.colorPicker : undefined),
                            // animationSpeed = colorPicker ? colorPicker.color.options.animationSpeed : 0
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
                    // prevent new eventListener for window
                    colorPickers.evt = true;
                    window[onOff]('mousedown', mousDownListener);
                }
            },
            // this is a way to prevent data binding on HTMLElements
            colorPickers = window.jsColorPicker.colorPickers || [],
            elms = document.querySelectorAll(selectors),
            testColors = new window.Colors({customBG: config.customBG, allMixDetails: true});

        window.jsColorPicker.colorPickers = colorPickers;

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

        return window.jsColorPicker.colorPickers;
    };

    global.ColorPicker.docCookies = function(key, val, options) {
        var encode = encodeURIComponent, decode = decodeURIComponent,
            cookies, n, tmp, cache = {},
            days;

        if (val === undefined) { // all about reading cookies
            cookies = document.cookie.split(/;\s*/) || [];
            for (n = cookies.length; n--; ) {
                tmp = cookies[n].split('=');
                if (tmp[0]) cache[decode(tmp.shift())] = decode(tmp.join('=')); // there might be '='s in the value...
            }

            if (!key) return cache; // return Json for easy access to all cookies
            else return cache[key]; // easy access to cookies from here
        } else { // write/delete cookie
            options = options || {};

            if (val === '' || options.expires < 0) { // prepare deleteing the cookie
                options.expires = -1;
                // options.path = options.domain = options.secure = undefined; // to make shure the cookie gets deleted...
            }

            if (options.expires !== undefined) { // prepare date if any
                days = new Date();
                days.setDate(days.getDate() + options.expires);
            }

            document.cookie = encode(key) + '=' + encode(val) +
                (days            ? '; expires=' + days.toUTCString() : '') +
                (options.path    ? '; path='    + options.path       : '') +
                (options.domain  ? '; domain='  + options.domain     : '') +
                (options.secure  ? '; secure'                        : '');
        }
    };
})(typeof window !== 'undefined' ? window : this);

if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports = jsColorPicker;
} else {
    window.jsColorPicker = jsColorPicker;
}