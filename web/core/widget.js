/**
 * File: widget.js
 * Project: Jovimetrix
 */

import { app } from "/scripts/app.js";

const _id = "jov.widgets.js";
const newTypes = ['RGB', 'FLOAT2', 'FLOAT3', 'FLOAT4', 'INTEGER2', 'INTEGER3', 'INTEGER4']
const PICKER_DEFAULT = '#ff0000';

const RGBWidget = (key, val = PICKER_DEFAULT, compute = false) => {
    const widget = {};
    widget.name = key;
    widget.type = 'RGB';
    widget.value = val;
    widget.draw = function(ctx, node, widgetWidth, widgetY, height) {
        const hide = this.type !== 'RGB' && app.canvas.ds.scale > 0.5;
        if (hide) return;

        const border = 3;
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.fillRect(0, widgetY, widgetWidth, height);
        ctx.fillStyle = this.value;
        ctx.fillRect(border, widgetY + border, widgetWidth - border * 2, height - border * 2);

        const color = this.value.default || this.value;
        if (!color) return;
    };
    widget.mouse = function(e, pos, node) {
        if (e.type === 'pointerdown') {
            const widgets = node.widgets.filter((w) => w.type === 'COLOR');
            for (const w of widgets) {
                // color picker
            }
        }
    };
    widget.computeSize = function(width) {
        return [width, 32];
    };
    return widget;
};

const SpinnerWidget = (app, type, labels, key, val, round=false, step=1) => {
    const offset = 4;
    const label_width = 50;
    const widget_padding = 15;
    const widget_padding2 = 2 * widget_padding;
    const label_full = widget_padding + label_width;

    let regions = [];
    let isDragging = -1;
    let startPosition = { x: 0, y: 0 };
    const widget = {
        name: key,
        type: type,
        value: val
    };

    widget.draw = function(ctx, node, width, Y, height) {
        const hide = this.type !== type && app.canvas.ds.scale > 0.5;
        if (hide) return;
        const data = [];
        labels.forEach((lbl, idx) => {
            data.push({ label: lbl, value: this.value[idx] });
        });
        const element_width = (width - label_full - widget_padding2) / (data.length);

        ctx.save();
        ctx.beginPath();
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
        ctx.roundRect(widget_padding, Y, width - widget_padding2, height, 8);
        ctx.stroke();
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
        ctx.fillText(key, widget_padding + offset, Y + height / 2 + offset);
        let x = label_full;
        regions = [];
        data.forEach(entry => {
            ctx.save();
            ctx.beginPath();
            ctx.rect(x, Y, element_width, height);
            ctx.clip();
            ctx.fillRect(x - 1, Y, 2, height);
            const size = entry.value.toString().length;
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
            ctx.fillText(entry.value, x + element_width / 2 - (size * 3.2), Y + height / 2 + offset);
            ctx.restore();
            regions.push([x + 1 - element_width, x - 1]);
            x += element_width;
        });
        ctx.restore();
    };

    widget.mouse = function (e, pos, node) {
        if (e.type === 'pointerdown' && pos[0] > label_full) {
            const x = pos[0] - label_full;
            const element_width = (node.size[0] - label_full - widget_padding2) / regions.length;
            const index = Math.floor(x / element_width);
            if (index >= 0 && index < regions.length) {
                isDragging = index;
                startPosition = { x: e.clientX, y: e.clientY };
            }
        } else if (isDragging > -1 && e.type === 'pointermove' && pos[0] > label_full) {
            widget.value[isDragging] -= (step * Math.sign(e.deltaY));
            startPosition = { x: e.clientX, y: e.clientY };
        } else if (isDragging > -1 && e.type === 'pointerup' && pos[0] > label_full) {
            isDragging = -1;
        }
    };

    widget.computeSize = function (width) {
        return [width, 20];
    };
    return widget;
};

const widgets = {
    name: _id,
    async getCustomWidgets(app) {
        return {
            RGB: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(RGBWidget(inputName, inputData[1]?.default || PICKER_DEFAULT)),
                minWidth: 35,
                minHeight: 35,
            }),
            INTEGER2: (node, inputName, data, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, "INTEGER2", ["x", "y"], inputName, data[1]?.default || [0, 0])),
                minWidth: node.size[0],
                minHeight: node.size[1],
            }),
            FLOAT2: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, "FLOAT2", ["x", "y"], inputName, inputData[1]?.default || [0, 0])),
                minWidth: node.size[0],
                minHeight: node.size[1],
            }),
            INTEGER3: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, "INTEGER3", ["x", "y", "z"], inputName, inputData[1]?.default || [0, 0, 0])),
                minWidth: node.size[0],
                minHeight: node.size[1],
            }),
            FLOAT3: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, "FLOAT3", ["x", "y", "z"], inputName, inputData[1]?.default || [0, 0, 0])),
                minWidth: node.size[0],
                minHeight: node.size[1],
            }),
            INTEGER4: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, "INTEGER4", ["x", "y", "z", "w"], inputName, inputData[1]?.default || [0, 0, 0, 0])),
                minWidth: node.size[0],
                minHeight: node.size[1],
            }),
            FLOAT4: (node, inputName, inputData, app) => ({
                widget: node.addCustomWidget(SpinnerWidget(app, "FLOAT4", ["x", "y", "z", "w"], inputName, inputData[1]?.default || [0, 0, 0, 0])),
                minWidth: node.size[0],
                minHeight: node.size[1],
            }),
        };
    }
};

app.registerExtension(widgets);
