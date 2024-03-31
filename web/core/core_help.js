/**
 * File: core_help.js
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js"
import { CONFIG_USER } from '../util/util_config.js'
import { node_cleanup } from '../util/util.js'
import "../extern/shodown.min.js"

const JOV_WEBWIKI_URL = "https://github.com/Amorano/Jovimetrix/wiki";

const TOOLTIP_LENGTH = 155;
const TOOLTIP_WIDTH_MAX = 225;
const TOOLTIP_WIDTH_OFFSET = 10;
const TOOLTIP_HEIGHT_MAX = LiteGraph.NODE_SLOT_HEIGHT * 0.65;
const FONT_SIZE = 7;

const dataCache = {};

const create_documentation_stylesheet = () => {
    const tag = 'mtb-documentation-stylesheet'
    let styleTag = document.head.querySelector(tag)
    if (styleTag) {
        return;
    }
    styleTag = document.createElement('style')
    styleTag.type = 'text/css'
    styleTag.id = tag

    styleTag.innerHTML = `
    .documentation-popup {
        background: var(--bg-color);
        position: absolute;
        color: var(--fg-color);
        font: 12px monospace;
        line-height: 1.25em;
        padding: 2px;
        border-radius: 3px;
        pointer-events: "inherit";
        z-index: 11111111;
        overflow:scroll;
    }
    .documentation-popup img {
        max-width: 100%;
    }
    .documentation-popup table {
        border-collapse: collapse;
        border: 1px var(--border-color) solid;
    }
    .documentation-popup th,
    .documentation-popup td {
        border: 1px var(--border-color) solid;
    }
    .documentation-popup th {
        background-color: var(--comfy-input-bg);
    }
        `
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
	},
    handleKeydown(e) {
        if (e.ctrlKey && e.shiftKey) {
            this.tooltips_visible = true;
        } else {
            this.tooltips_visible = false;
        };
        console.log(this)
    },
	beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.category?.startsWith("JOVIMETRIX")) {
            return;
        }

        const self = this;
        let opts = { icon_size: 14, icon_margin: 3 }
        const iconSize = opts.icon_size ? opts.icon_size : 14;
        const iconMargin = opts.icon_margin ? opts.icon_margin : 3;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            this.docElement = null;
            this.offsetX = 0;
            this.offsetY = 0;
            this.show_doc = false;
            this.onRemoved = function () {
                node_cleanup(this);
                if (this.docElement) {
                    this.docElement.parentNode.removeChild(this.docElement)
                    this.docElement = null;
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
                        if (!item.hidden || item.hidden == false) {
                            const data = {
                                name: item.name,
                                y: offset
                            }
                            offset += TOOLTIP_HEIGHT_MAX * 1.45;
                            visible.push(data);
                        }
                    };
                    for(const item of this.widgets || []) {
                        if (!item.hidden || item.hidden == false) {
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
            if (this.show_doc && this.docElement === null) {
                create_documentation_stylesheet();
                this.docElement = document.createElement('div');
                this.docElement.classList.add('documentation-popup');
                if (!(nodeData.name in dataCache)) {
                    // Load data from URL asynchronously if it ends with .md
                    if (nodeData.description.endsWith('.md')) {
                        // Check if data is already cached
                        // Fetch data from URL
                        fetch(nodeData.description)
                            .then(response => {
                                if (!response.ok) {
                                    this.docElement.innerHTML = `Failed to load documentation\n\n${response}`
                                }
                                return response.text();
                            })
                            .then(data => {
                                // Cache the fetched data
                                dataCache[nodeData.name] = documentationConverter.makeHtml(data);
                                this.docElement.innerHTML = dataCache[nodeData.name];
                            })
                            .catch(error => {
                                this.docElement.innerHTML = `Failed to load documentation\n\n${error}`
                                console.error('Error:', error);
                            });
                    } else {
                        // If description does not end with .md, set data directly
                        dataCache[nodeData.name] = documentationConverter.makeHtml(dataCache[nodeData.name]);
                        this.docElement.innerHTML = dataCache[nodeData.name];
                    }
                } else {
                    this.docElement.innerHTML = dataCache[nodeData.name];
                }
                document.body.appendChild(this.docElement)
            }

            if (!this.show_doc && this.docElement !== null) {
                this.docElement.parentNode.removeChild(this.docElement)
                this.docElement = null;
                return
            }

            if (this.show_doc && this.docElement !== null) {
                const rect = ctx.canvas.getBoundingClientRect()
                const scaleX = rect.width / ctx.canvas.width
                const scaleY = rect.height / ctx.canvas.height
                const transform = new DOMMatrix()
                    .scaleSelf(scaleX, scaleY)
                    .multiplySelf(ctx.getTransform())
                    .translateSelf(this.size[0] + 10, -32)

                const width = Math.min(512,  2 * this.size[0] - LiteGraph.NODE_MIN_WIDTH);
                const height = (this.size[1] || this.parent?.inputHeight || 0) + 48;
                const scale = new DOMMatrix().scaleSelf(transform.a, transform.d);
                Object.assign(this.docElement.style, {
                    transformOrigin: '0 0',
                    transform: scale,
                    left: `${transform.a + transform.e}px`,
                    top: `${transform.d + transform.f - 8}px`,
                    width: `${width}px`,
                    height: `${height}px`,
                })
            }

            ctx.save()
            ctx.translate(x, iconSize - 34) // Position the icon on the canvas
            ctx.scale(iconSize / 32, iconSize / 32) // Scale the icon to the desired size
            ctx.strokeStyle = 'rgba(255,255,255,0.3)'
            ctx.lineCap = 'round'
            ctx.lineJoin = 'round'
            ctx.lineWidth = 2.4
            // ctx.stroke(questionMark);
            ctx.font = '36px monospace'
            ctx.fillText('?', 0, 24)
            ctx.restore()
            return me;
        }

        // HELP!
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const me = getExtraMenuOptions?.apply(this, arguments);
            const widget_tooltip = (this.widgets || [])
                .find(widget => widget.type === 'JTOOLTIP');
            if (!widget_tooltip) {
                return me;
            }
            const tips = widget_tooltip.options.default || {};
            const url = tips['_'];
            if (url === undefined) {
                return me;
            }
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
                return true; // Return true to indicate the event was handled
            }
            return r;
        }
	}
})
