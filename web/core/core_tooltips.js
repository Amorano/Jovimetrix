/**
 * File: core_tooltips.js
 * Project: Jovimetrix
 */

import { app } from "../../../scripts/app.js"

import { CONFIG_USER } from '../util/util_config.js'
// const TOOLTIP_COLOR = CONFIG_USER.color.tooltips;

const TOOLTIP_LENGTH = 155;
const TOOLTIP_WIDTH_MAX = 225;
const TOOLTIP_WIDTH_OFFSET = 10;
const TOOLTIP_HEIGHT_MAX = LiteGraph.NODE_SLOT_HEIGHT * 0.65;
const FONT_SIZE = 7;

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
    name: "jovimetrix.tooltips",
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
    },
	beforeRegisterNodeDef(nodeType) {
        const self = this;
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const me = onDrawForeground?.apply?.(this, arguments);
            if (!self.tooltips_visible || this.flags.collapsed) {
                return me;
            }
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
            return me;
        }

        // HELP!
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const self = this;
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
                    window.open(url, '_blank');
                    self.setDirtyCanvas(true, true);
                }
            }];
            if (help_menu.length) {
                options.push(...help_menu, null);
            }
            return me;
        }
	}
})
