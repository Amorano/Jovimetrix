/**
 * File: core_tooltips.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js"

import { CONFIG_USER } from '../util/util_config.js'
// const TOOLTIP_COLOR = CONFIG_USER.color.tooltips;

const TOOLTIP_LENGTH = 175;
const TOOLTIP_LINES_MAX = 3;
const TOOLTIP_WIDTH_MAX = 200;
const TOOLTIP_WIDTH_OFFSET = 10;
const Y_OFFSET_MAX = LiteGraph.NODE_SLOT_HEIGHT * 0.70;

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
    return lines.join('\n');
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
                let offset = Y_OFFSET_MAX * 0.4;
                for(const item of this.inputs || []) {
                    if (!item.hidden || item.hidden == false) {
                        const data = {
                            name: item.name,
                            y: offset
                        }
                        offset += Y_OFFSET_MAX * 1.35;
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
                ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR + alpha;
                const offset_y = visible[0].y;
                const height = this.size[1] - offset_y;
                ctx.roundRect(-TOOLTIP_WIDTH_MAX-TOOLTIP_WIDTH_OFFSET,
                    offset_y,
                    TOOLTIP_WIDTH_MAX + 4, height, 8);
                ctx.fill();
                ctx.fillStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
                ctx.roundRect(-TOOLTIP_WIDTH_MAX-TOOLTIP_WIDTH_OFFSET,
                    offset_y,
                    TOOLTIP_WIDTH_MAX + 4, height, 8);
                ctx.stroke()
                let index = 0;
                for(const item of visible) {
                    let text = tips[item.name] || item.options?.tooltip || item.name;
                    text = text.slice(0, TOOLTIP_LENGTH);
                    const factor = text.length < 58 ? 42 : 58;
                    text = wrapText(text, factor)
                    var lines = text.split('\n').slice(0, TOOLTIP_LINES_MAX);
                    const tooltip_line_count = Math.min(TOOLTIP_LINES_MAX, lines.length);
                    let font_size = LiteGraph.NODE_SUBTEXT_SIZE / tooltip_line_count;
                    font_size = Math.max(8, font_size - (text.length / 25));
                    ctx.font = `${font_size-1}px sans-serif`;
                    ctx.fillStyle = TOOLTIP_COLOR.slice(0, 7) + alpha;
                    const scalar = tooltip_line_count > 2 ? 2 : 1;
                    var offset_tip = (item?.y || item.last_y) + scalar * Y_OFFSET_MAX / tooltip_line_count;
                    const offset_tip_step = (Y_OFFSET_MAX / tooltip_line_count + Y_OFFSET_MAX * (tooltip_line_count-1) / tooltip_line_count) * 0.75;
                    for (const line of lines) {
                        const sz = ctx.measureText(line);
                        const left = Math.max(-TOOLTIP_WIDTH_MAX, -TOOLTIP_WIDTH_OFFSET-sz.width);
                        ctx.fillText(line, left, offset_tip);
                        offset_tip += offset_tip_step;
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
