const ColorPicker = {
    colorPickers: window.jsColorPicker?.colorPickers || [],
    docCookies: {
        getItem: (key, def) => {
            const data = local_get(key, def);
            return data;
        },
        setItem: (key, value) => {
            local_set(key, value);
        }
    },

    createInstance: (elm, config) => {
        const initConfig = {
            input: elm,
            patch: elm,
            isIE8: !!document.all && !document.addEventListener,
            margin: { left: -1, top: 2 },
            customBG: '#FFFFFF',
            color: ColorPicker.extractValue(elm),
            initStyle: 'display: none',
            mode: ColorPicker.docCookies('colorPickerMode') || 'hsv-h',
            memoryColors: ColorPicker.docCookies('colorPickerMemos' + ((config || {}).noAlpha ? 'NoAlpha' : '')),
            size: ColorPicker.docCookies('colorPickerSize') || 1,
            renderCallback: ColorPicker.renderCallback,
            actionCallback: ColorPicker.actionCallback
        };

        for (const n in config) {
            initConfig[n] = config[n];
        }

        return new ColorPicker(initConfig);
    },

    renderCallback: (colors, mode) => {
        const rgb = Object.values(colors.RND.rgb).reverse();
        const AHEX = !colors.HEX.includes("NAN") ? colorRGB2Hex(rgb) : "#353535FF";

        this.patch.style.cssText =
            'color:' + (colors.rgbaMixCustom.luminance > 0.22 ? '#222' : '#ddd') + ';' +
            'background-color: ' + AHEX + ';';

        this.input.setAttribute("color", AHEX);

        if (this.displayCallback) {
            this.displayCallback(colors, mode, this);
        }
    },

    extractValue: (elm) => {
        const val = elm.getAttribute('color') || elm.style.backgroundColor || '#353535FF';
        return val.includes("NAN") ? "#353535FF" : val;
    },

    actionCallback: (event, action) => {
        if (action === 'toMemory') {
            const memos = this.nodes.memos;
            const cookieTXT = [];

            for (let n = 0, m = memos.length; n < m; n++) {
                let backgroundColor = memos[n].style.backgroundColor;
                let opacity = memos[n].style.opacity;
                opacity = Math.round((opacity === '' ? 1 : opacity) * 100) / 100;
                cookieTXT.push(backgroundColor.replace(/, /g, ',').replace('rgb(', 'rgba(').replace(')', ',' + opacity + ')'));
            }

            ColorPicker.docCookies('colorPickerMemos' + (this.noAlpha ? 'NoAlpha' : ''), "'" + cookieTXT.join("','") + "'");
        } else if (action === 'resizeApp') {
            ColorPicker.docCookies('colorPickerSize', this.color.options.currentSize);
        } else if (action === 'modeChange') {
            const mode = this.color.options.mode;
            ColorPicker.docCookies('colorPickerMode', mode.type + '-' + mode.z);
        }
    },

    doEventListeners: (elm, multiple, off, elms) => {
        const onOff = off ? 'removeEventListener' : 'addEventListener';
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
                'position: absolute;' + (!ColorPicker.colorPickers[index].cssIsReady ? 'display: none;' : '') +
                'left:' + (position.left + options.margin.left - atrect.left) + 'px;' +
                'top:' + (position.top + +this.offsetHeight + options.margin.top - atrect.top) + 'px;';

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
                    colorPickerUI.style.display = 'block';
                }
            }, 10);
        };

        elm[onOff]('focus', focusListener);

        if (!ColorPicker.colorPickers.evt || off) {
            ColorPicker.colorPickers.evt = true;
            window[onOff]('mousedown', (e) => {
                const colorPicker = ColorPicker.colorPickers.current;
                const colorPickerUI = (colorPicker ? colorPicker.nodes.colorPicker : undefined);
                const isColorPicker = colorPicker && (function(elm) {
                    while (elm) {
                        if ((elm.className || '').indexOf('cp-app') !== -1) return elm;
                        elm = elm.parentNode;
                    }
                    return false;
                })(e.target);

                if (isColorPicker && Array.prototype.indexOf.call(ColorPicker.colorPickers, isColorPicker)) {
                    if (e.target === colorPicker.nodes.exit) {
                        colorPickerUI.style.display = 'none';
                        document.activeElement.blur();
                    }
                } else if (Array.prototype.indexOf.call(elms, e.target) !== -1) {
                } else if (colorPickerUI) {
                    colorPickerUI.style.display = 'none';
                }
            });
        }
    }
};

export function colorPicker(selectors, config, callback, elms) {
    const testColors = new window.Colors({ customBG: config.customBG, allMixDetails: true });

    for (let n = 0, m = elms.length; n < m; n++) {
        const elm = elms[n];

        if (config === 'destroy') {
            ColorPicker.doEventListeners(elm, (config && config.multipleInstances), true, elms);
            if (ColorPicker.colorPickers[n]) {
                ColorPicker.colorPickers[n].destroyAll();
            }
        } else {
            const color = ColorPicker.extractValue(elm);
            const value = color.split('(');

            testColors.setColor(color);
            if (config && config.init) {
                config.init(elm, testColors.colors);
            }
            elm.setAttribute('data-colorMode', value[1] ? value[0].substr(0, 3) : 'HEX');
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
