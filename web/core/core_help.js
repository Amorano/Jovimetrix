/**
 * File: core_help.js
 * Project: Jovimetrix
 * code based on mtb nodes by Mel Massadian https://github.com/melMass/comfy_mtb/
 */

import { app } from "../../../scripts/app.js"
import { CONVERTED_TYPE } from '../util/util_widget.js'
import { CONFIG_USER } from '../util/util_config.js'
import { node_cleanup } from '../util/util.js'
import "../extern/shodown.min.js"

const JOV_WEBWIKI_URL = "https://github.com/Amorano/Jovimetrix/wiki";
const JOV_HELP_URL = "https://raw.githubusercontent.com/Amorano/Jovimetrix-examples/master";

const TOOLTIP_LENGTH = 155;
const TOOLTIP_WIDTH_MAX = 225;
const TOOLTIP_WIDTH_OFFSET = 10;
const TOOLTIP_HEIGHT_MAX = LiteGraph.NODE_SLOT_HEIGHT * 0.65;
const FONT_SIZE = 7;

const dataCache = {};

const create_documentation_stylesheet = () => {
    const tag = 'jov-documentation-stylesheet'
    let styleTag = document.head.querySelector(tag)
    if (styleTag) {
        return;
    }
    styleTag = document.createElement('style')
    styleTag.type = 'text/css'
    styleTag.id = tag

    styleTag.innerHTML = `
    .jov-documentation-popup {
        background: var(--bg-color);
        position: absolute;
        color: var(--fg-color);
        font: 10px monospace;
        line-height: 1.25em;
        padding: 2px;
        border-radius: 7px;
        pointer-events: "inherit";
        border-style: solid;
        border-width: medium;
        border-color: var(--border-color);
        z-index: 25;
        overflow: hidden;
        width: 315px;
        height: 295px;
        min-width: 315px;
        min-height: 65px;
    }
    .jov-documentation-popup img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 90%;
    }
    .jov-documentation-popup table {
        border-collapse: collapse;
        border: 1px var(--border-color) solid;
    }
    .jov-documentation-popup th,
    .jov-documentation-popup td {
        border: 1px var(--border-color) solid;
    }
    .jov-documentation-popup th {
        background-color: var(--comfy-input-bg);
    }
    .content-wrapper {
        overflow: auto;
        max-height: 100%;
        /* Scrollbar styling for Chrome */
        &::-webkit-scrollbar {
           width: 6px;
        }
        &::-webkit-scrollbar-track {
           background: var(--bg-color);
        }
        &::-webkit-scrollbar-thumb {
           background-color: var(--fg-color);
           border-radius: 6px;
           border: 3px solid var(--bg-color);
        }

        /* Scrollbar styling for Firefox */
        scrollbar-width: thin;
        scrollbar-color: var(--fg-color) var(--bg-color);
        a {
          color: yellow;
        }
        a:visited {
          color: orange;
        }
        a:hover {
          color: red;
        }
    }`
    document.head.appendChild(styleTag)
}

const documentationConverter = new showdown.Converter({
    tables: true,
    strikethrough: true,
    emoji: true,
    ghCodeBlocks: true,
    tasklists: true,
    ghMentions: true,
    smoothLivePreview: true,
    simplifiedAutoLink: true,
    parseImgDimensions: true,
    openLinksInNewWindow: true,
});

/*
* wraps a single text line into maxWidth chunks
*/
function wrapText(text, maxWidth=145) {
    let words = text.split(' ');
    let lines = [];
    let currentLine = '';
    words.forEach(word => {
        let potentialLine = currentLine + ' ' + word;
        if (potentialLine.trim().length <= maxWidth) {
            currentLine = potentialLine.trim();
        } else {
            lines.push(currentLine);
            currentLine = word;
        }
    });
    lines.push(currentLine);
    return lines;
}

export const JTooltipWidget = (app, name, opts) => {
    let options = opts || {};
    options.serialize = false;
    const w = {
        name: name,
        type: "JTOOLTIP",
        hidden: true,
        options: options,
        draw: function (ctx, node, width, Y, height) {
            return;
        },
        computeSize: function (width) {
            return [width, 0];
        }
    }
    return w
}

app.registerExtension({
    name: "jovimetrix.help",
    async getCustomWidgets(app) {
        return {
            JTOOLTIP: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(JTooltipWidget(app, inputName, inputData[1]))
            })
        }
    },
    init() {
        this.tooltips_visible = false;
        window.addEventListener("keydown", (e) => {
            this.handleKeydown(e);
        });
        window.addEventListener("keyup", (e) => {
            this.handleKeydown(e);
        });
        create_documentation_stylesheet();
	},
    handleKeydown(e) {
        if ((e.altKey && e.shiftKey) || (e.ctrlKey && e.shiftKey)) {
            this.tooltips_visible = true;
        } else {
            this.tooltips_visible = false;
        };
    },
	beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.category?.startsWith("JOVIMETRIX")) {
            return;
        }

        const self = this;
        let opts = { icon_size: 14, icon_margin: 3 }
        const iconSize = opts.icon_size ? opts.icon_size : 14;
        const iconMargin = opts.icon_margin ? opts.icon_margin : 3;
        let docElement = null;
        let contentWrapper = null;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            this.offsetX = 0;
            this.offsetY = 0;
            this.show_doc = false;
            this.onRemoved = function () {
                node_cleanup(this);
                if (docElement) {
                    docElement.remove();
                    docElement = null;
                }
                if (contentWrapper) {
                    contentWrapper.remove();
                    contentWrapper = null;
                }
            }
            return me;
        }

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const me = onDrawForeground?.apply?.(this, arguments);
            if (this.flags.collapsed) return me;
            if (self.tooltips_visible) {
                const TOOLTIP_COLOR = CONFIG_USER.color.tooltips;
                let alpha = TOOLTIP_COLOR.length > 6 ? TOOLTIP_COLOR.slice(-2) : "FF";
                for (const selectedNode of Object.values(app.canvas.selected_nodes)) {
                    if (selectedNode !== this) {
                        continue
                    }
                    const widget_tooltip = (selectedNode.widgets || [])
                        .find(widget => widget.type === 'JTOOLTIP');
                    if (!widget_tooltip) {
                        continue;
                    }
                    const tips = widget_tooltip.options.default || {};
                    let visible = [];
                    let offset = TOOLTIP_HEIGHT_MAX / 4;
                    for(const item of this.inputs || []) {
                        if (item.hidden == undefined || item?.hidden == false) {
                            const data = {
                                name: item.name,
                                y: offset
                            }
                            visible.push(data);
                        }
                        offset += TOOLTIP_HEIGHT_MAX * 1.45;
                    };
                    for(const item of this.widgets || []) {
                        if ((item.type != CONVERTED_TYPE) && (item.hidden == undefined || item?.hidden == false)) {
                            const data = {
                                name: item.name,
                                options: item.options,
                                y: item.y || item.last_y ||
                                    visible[visible.length-1]?.y ||
                                    visible[visible.length-1]?.last_y
                            }
                            visible.push(data);
                        }
                    };
                    if (visible.length == 0) {
                        continue;
                    }
                    ctx.save();
                    ctx.lineWidth = 1
                    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR; //"#333333" + alpha; // LiteGraph.WIDGET_BGCOLOR
                    const offset_y = visible[0].y;
                    const height = this.size[1] - offset_y;
                    ctx.roundRect(-TOOLTIP_WIDTH_MAX-TOOLTIP_WIDTH_OFFSET,
                        offset_y - TOOLTIP_HEIGHT_MAX / 2,
                        TOOLTIP_WIDTH_MAX + 4, height, 8);
                    ctx.fill();
                    ctx.fillStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
                    ctx.roundRect(-TOOLTIP_WIDTH_MAX-TOOLTIP_WIDTH_OFFSET,
                        offset_y- TOOLTIP_HEIGHT_MAX / 2,
                        TOOLTIP_WIDTH_MAX + 4, height, 8);
                    ctx.stroke();

                    let index = 0;
                    for(const item of visible) {
                        let text = tips[item.name] || item.options?.tooltip || item.name;
                        text = text.slice(0, TOOLTIP_LENGTH).toUpperCase();
                        const size = text.length;
                        let wrap = 50;
                        if (size > 45) {
                            wrap = 50;
                        } else if (size > 80) {
                            wrap = 52;
                        }
                        var lines = wrapText(text, wrap).slice(0, 3);
                        ctx.font = FONT_SIZE - lines.length / 2 + "px sans-serif";
                        ctx.fillStyle = TOOLTIP_COLOR.slice(0, 7) + alpha;
                        const offset = TOOLTIP_HEIGHT_MAX * 2 / (lines.length+1);
                        let idx = 1;
                        for (const line of lines) {
                            const sz = ctx.measureText(line);
                            const left = Math.max(-TOOLTIP_WIDTH_MAX, -TOOLTIP_WIDTH_OFFSET-sz.width);
                            ctx.fillText(line, left, item.y + idx * offset);
                            idx += 1;
                        }
                        index += 1
                    }
                    ctx.restore();
                }
            }

            const x = this.size[0] - iconSize - iconMargin
            if (this.show_doc && docElement === null) {
                docElement = document.createElement('div')
                contentWrapper = document.createElement('div');
                docElement.appendChild(contentWrapper);
                contentWrapper.classList.add('content-wrapper');
                docElement.classList.add('jov-documentation-popup');
                if (!(nodeData.name in dataCache)) {
                    // Load data from URL asynchronously if it ends with .md
                    const widget_tooltip = (this.widgets || [])
                        .find(widget => widget.type === 'JTOOLTIP');
                    if (widget_tooltip) {
                        const tips = widget_tooltip.options.default || {};
                        let url = tips['*'];
                        if (url.endsWith('.md')) {
                            url = `${JOV_HELP_URL}/${url}`;
                            console.log(url)
                            // Check if data is already cached
                            // Fetch data from URL
                            fetch(url)
                                .then(response => {
                                    if (!response.ok) {
                                        contentWrapper.innerHTML = `Failed to load documentation\n\n${response}`
                                    }
                                    return response.text();
                                })
                                .then(data => {
                                    // Cache the fetched data
                                    dataCache[nodeData.name] = documentationConverter.makeHtml(data);
                                    contentWrapper.innerHTML = dataCache[nodeData.name];
                                })
                                .catch(error => {
                                    contentWrapper.innerHTML = `Failed to load documentation\n\n${error}`
                                    console.error('Error:', error);
                                });
                        } else {
                            // If description does not end with .md, set data directly
                            dataCache[nodeData.name] = documentationConverter.makeHtml(dataCache[nodeData.name]);
                            contentWrapper.innerHTML = dataCache[nodeData.name];
                        }
                    }
                } else {
                    contentWrapper.innerHTML = dataCache[nodeData.name];
                }

                // resize handle
                const resizeHandle = document.createElement('div');
                resizeHandle.style.width = '0';
                resizeHandle.style.height = '0';
                resizeHandle.style.position = 'absolute';
                resizeHandle.style.bottom = '0';
                resizeHandle.style.right = '0';
                resizeHandle.style.cursor = 'se-resize';
                const borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border-color').trim();
                resizeHandle.style.borderTop = '5px solid transparent';
                resizeHandle.style.borderLeft = '5px solid transparent';
                resizeHandle.style.borderBottom = `5px solid ${borderColor}`;
                resizeHandle.style.borderRight = `5px solid ${borderColor}`;
                docElement.appendChild(resizeHandle)
                let isResizing = false
                let startX, startY, startWidth, startHeight

                resizeHandle.addEventListener('mousedown', function (e) {
                        e.preventDefault();
                        e.stopPropagation();
                        isResizing = true;
                        startX = e.clientX;
                        startY = e.clientY;
                        startWidth = parseInt(document.defaultView.getComputedStyle(docElement).width, 10);
                        startHeight = parseInt(document.defaultView.getComputedStyle(docElement).height, 10);
                    },
                    { signal: this.helpClose.signal },
                );

                const buttons = document.createElement('div');
                // wiki page
                const wikiButton = document.createElement('div');
                wikiButton.textContent = '🌐';
                wikiButton.style.position = 'absolute';
                wikiButton.style.top = '6px';
                wikiButton.style.right = '22px';
                wikiButton.style.cursor = 'pointer';
                wikiButton.style.padding = '4px';
                wikiButton.style.font = 'bold 14px monospace';
                wikiButton.addEventListener('mousedown', (e) => {
                    e.stopPropagation();
                    const widget_tooltip = (this.widgets || [])
                        .find(widget => widget.type === 'JTOOLTIP');
                    if (widget_tooltip) {
                        const tips = widget_tooltip.options.default || {};
                        const url = tips['_'];
                        if (url !== undefined) {
                            window.open(`${JOV_WEBWIKI_URL}/${url}`, '_blank');
                        }
                        wikiButton.tooltip = `WIKI PAGE: ${url}`
                    }
                });
                buttons.appendChild(wikiButton);

                // close button
                const closeButton = document.createElement('div');
                closeButton.textContent = '❌';
                closeButton.style.position = 'absolute';
                closeButton.style.top = '6px';
                closeButton.style.right = '4px';
                closeButton.style.cursor = 'pointer';
                closeButton.style.padding = '4px';
                closeButton.style.font = 'bold 14px monospace';
                closeButton.addEventListener('mousedown', (e) => {
                        e.stopPropagation();
                        this.show_doc = !this.show_doc
                        docElement.parentNode.removeChild(docElement)
                        docElement = null
                        if (contentWrapper) {
                            contentWrapper.remove()
                            contentWrapper = null
                        }
                    },
                    { signal: this.helpClose.signal },
                );
                buttons.appendChild(closeButton);
                docElement.appendChild(buttons);

                document.addEventListener('mousemove', function (e) {
                    if (!isResizing) return;
                    const scale = app.canvas.ds.scale;
                    const newWidth = startWidth + (e.clientX - startX) / scale;
                    const newHeight = startHeight + (e.clientY - startY) / scale;;
                    docElement.style.width = `${newWidth}px`;
                    docElement.style.height = `${newHeight}px`;
                    },
                    { signal: this.helpClose.signal },
                );
                document.addEventListener('mouseup', function () {
                        isResizing = false
                    },
                    { signal: this.helpClose.signal },
                );
                document.body.appendChild(docElement);
            } else if (!this.show_doc && docElement !== null) {
                docElement.parentNode.removeChild(docElement)
                docElement = null;
            }
            if (this.show_doc && docElement !== null) {
                const rect = ctx.canvas.getBoundingClientRect()
                const scaleX = rect.width / ctx.canvas.width
                const scaleY = rect.height / ctx.canvas.height
                const transform = new DOMMatrix()
                    .scaleSelf(scaleX, scaleY)
                    .multiplySelf(ctx.getTransform())
                    .translateSelf(this.size[0] * scaleX * Math.max(1.0,window.devicePixelRatio) , 0)
                    .translateSelf(10, -32)

                const scale = new DOMMatrix()
                    .scaleSelf(transform.a, transform.d);

                const styleObject = {
                    transformOrigin: '0 0',
                    transform: scale,
                    left: `${transform.a + transform.e}px`,
                    top: `${transform.d + transform.f}px`,
                };
                Object.assign(docElement.style, styleObject);
            }

            ctx.save()
            ctx.translate(x, iconSize - 35) // Position the icon on the canvas
            ctx.scale(iconSize / 32, iconSize / 32) // Scale the icon to the desired size
            ctx.font = 'bold 52px monospace'
            ctx.fillStyle = 'rgb(255,120,240,0.50)';
            ctx.fillText('?', 0, 32);
            ctx.translate(1, 3) // Position the icon on the canvas
            ctx.font = 'bold 49px monospace'
            ctx.fillStyle = 'rgb(40,10,20,0.70)';
            ctx.fillText('?', 0, 28);
            ctx.translate(1, 3) // Position the icon on the canvas
            ctx.fillStyle = 'rgb(250,80,250)';
            ctx.font = 'bold 46px monospace';
            ctx.fillText('?', 0, 24);
            ctx.restore()
            return me;
        }

        // MENU HELP!
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const me = getExtraMenuOptions?.apply(this, arguments);
            const widget_tooltip = (this.widgets || [])
                .find(widget => widget.type === 'JTOOLTIP');
            if (widget_tooltip) {
                const tips = widget_tooltip.options.default || {};
                const url = tips['_'];
                const help_menu = [{
                    content: `HELP: ${this.title}`,
                    callback: () => {
                        LiteGraph.closeAllContextMenus();
                        window.open(`${JOV_WEBWIKI_URL}/${url}`, '_blank');
                        this.setDirtyCanvas(true, true);
                    }
                }];
                if (help_menu.length) {
                    options.push(...help_menu, null);
                }
            }
            return me;
        }

        // ? clicked
        const mouseDown = nodeType.prototype.onMouseDown
        nodeType.prototype.onMouseDown = function (e, localPos, canvas) {
            const r = mouseDown ? mouseDown.apply(this, arguments) : undefined
            const iconX = this.size[0] - iconSize - iconMargin
            const iconY = iconSize - 34
            if (
                localPos[0] > iconX &&
                localPos[0] < iconX + iconSize &&
                localPos[1] > iconY &&
                localPos[1] < iconY + iconSize
            ) {
                // Pencil icon was clicked, open the editor
                // this.openEditorDialog();
                if (this.show_doc === undefined) {
                    this.show_doc = true;
                } else {
                    this.show_doc = !this.show_doc;
                }
                if (this.show_doc) {
                    this.helpClose = new AbortController()
                } else {
                    this.helpClose.abort()
                }
                return true;
            }
            return r;
        }
	}
})
