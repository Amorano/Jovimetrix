/**
 * File: core_experiment.js
 * Project: Jovimetrix
 *
 * crazy stuffs
 */

import { app } from "../../../scripts/app.js"

class Vec4ColorWidget {
    constructor(...args) {
        const [inputName, opts] = args

        this.name = inputName || 'Vec4Color'
        this.type = 'VEC4COLOR'
        this.selectedPointIndex = null
        this.options = opts
        this.value = this.value || [1.0, 1.0, 1.0, 1.0]

        this.hueSaturation = [null, null];
        this.brightness = 1.0;
        this.brightnessWidthPct = 0.1;

        this.size = [200, 200]
        this.wY = 0;
    }

    draw(ctx, node, widgetWidth, widgetY) {
        this.wY = widgetY;
        this.wWidth = widgetWidth;
        this.wHeight = node.size[1] - widgetY;

        let y = widgetY;

        let width = widgetWidth;
        let height = node.size[1];

        let barWidth = width * this.brightnessWidthPct;
        let barHeight = height - y;

        let diskWidth = width - barWidth;
        let diskHeight = height - y;

        let diskX = diskWidth * 0.5;
        let diskY = diskHeight * 0.5 + y;

        let diskRadiusX = diskWidth * 0.5 - 1.0;
        let diskRadiusY = diskHeight * 0.5 - 1.0;

        // generic function for drawing a canvas disc
        function drawDisk (ctx, coords, radius, steps, colorCallback) {
            let x = coords[0] || coords; // coordinate on x-axis
            let y = coords[1] || coords; // coordinate on y-axis
            let a = radius[0] || radius; // radius on x-axis
            let b = radius[1] || radius; // radius on y-axis
            let angle = 360;
            let coef = Math.PI / 180;

            ctx.save();
            ctx.translate(x - a, y - b);
            ctx.scale(a, b);

            steps = (angle / steps) || 360;

            for (; angle > 0 ; angle -= steps) {
                ctx.beginPath();
                if (steps !== 360) {
                    ctx.moveTo(1, 1); // stroke
                }
                ctx.arc(1, 1, 1,
                    (angle - (steps / 2) - 1) * coef,
                    (angle + (steps / 2) + 1) * coef);

                if (colorCallback) {
                    colorCallback(ctx, angle);
                }
                else {
                    ctx.fillStyle = 'black';
                    ctx.fill();
                }
            }
            ctx.restore();
        }

        // Draw HueSaturation Color Disk
        drawDisk(
            ctx,
            [diskX, diskY],
            [diskRadiusX, diskRadiusY],
            360,
            function (ctx, angle) {
                let gradient = ctx.createRadialGradient(1, 1, 1, 1, 1, 0);
                gradient.addColorStop(0, 'hsl(' + (360 - angle + 0) + ', 100%, 50%)');
                gradient.addColorStop(1, '#fff');

                ctx.fillStyle = gradient;
                ctx.fill();
            }
        );

        // Draw brightness bar
        let barX = width - barWidth;
        let barY = y;

        let gradient = ctx.createLinearGradient(0, 0, 0, barHeight);
        gradient.addColorStop(0, 'white');
        gradient.addColorStop(1, 'black');
        ctx.fillStyle = gradient;
        ctx.fillRect(barX, barY, barWidth, barHeight);

        // Draw brightness position (based this.brightness)
        // with two triangles at both size and a line in-between
        if (this.brightness !== null) {
            let bX = barX;
            let bY = y + barHeight - this.brightness * barHeight;
            let bColor = '#fff';
            ctx.strokeStyle = bColor;
            ctx.fillStyle = bColor;
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            ctx.moveTo(bX, bY);
            ctx.lineTo(bX + barWidth, bY);
            ctx.closePath();
            ctx.stroke();

            // Left triangle
            ctx.beginPath();
            ctx.moveTo(bX, bY);
            ctx.lineTo(bX - 5, bY + 5);
            ctx.lineTo(bX - 5, bY - 5);
            ctx.closePath();
            ctx.fill();

            // Right triangle
            ctx.beginPath();
            ctx.moveTo(bX + barWidth, bY);
            ctx.lineTo(bX + barWidth + 5, bY + 5);
            ctx.lineTo(bX + barWidth + 5, bY - 5);
            ctx.closePath();
            ctx.fill();
        }

        // Draw color position (based this.hueSaturation)
        // stored in polar coordinates
        if (this.hueSaturation[0] !== null && this.hueSaturation[1] !== null) {
            let angle = this.hueSaturation[0];
            let dist = this.hueSaturation[1];
            let hX = -Math.cos(angle) * dist * diskRadiusX + diskX;
            let hY = Math.sin(angle) * dist * diskRadiusY + diskY;

            ctx.fillStyle = '#000';
            ctx.beginPath();
            let radius = 4;
            ctx.arc(hX, hY, radius, 0, 2 * Math.PI, false);
            ctx.fill();

            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            radius = 8;
            ctx.arc(hX, hY, radius, 0, 2 * Math.PI, false);
            ctx.stroke();
        }

        // Draw Selected Color (based this.value)
        // as a rectangle with the color un the top left corner
        let cW = this.wWidth - 50;
        let cH = this.wY;

        ctx.fillStyle = 'rgba(' + Math.round(this.value[0] * 255) + ',' + Math.round(this.value[1] * 255) + ',' + Math.round(this.value[2] * 255) + ',' + this.value[3] + ')';
        ctx.fillRect(0, 0, cW, cH);
    }

    mouse(e, pos, node) {
        if (e.type === 'pointermove') {
            // if pos is inside the brightness bar change brightness based on Y position
            if (pos[0] > this.wWidth - this.wWidth * this.brightnessWidthPct) {
                this.brightness = 1.0 - (pos[1] - this.wY) / this.wHeight;
                this.brightness = Math.min(1.0, Math.max(0.0, this.brightness));
            }
            else {
                let barWidth = this.wWidth * this.brightnessWidthPct;
                let diskWidth = this.wWidth - barWidth;
                let diskHeight = this.wHeight - this.wY;
                let diskX = diskWidth * 0.5;
                let diskY = diskHeight * 0.5 + this.wY;
                let diskRadiusX = diskWidth * 0.5 - 1.0;
                let diskRadiusY = diskHeight * 0.5 - 1.0;

                let x = pos[0];
                let y = pos[1];
                // convert to polar coordinates
                let dx = x - diskX;
                let dy = y - diskY;
                let angle = Math.atan2(dy, -dx);
                let dist = Math.sqrt(dx * dx + dy * dy);
                let maxDist = Math.min(diskRadiusX, diskRadiusY);
                dist = Math.min(dist, maxDist);
                angle = angle;
                this.hueSaturation = [ angle,  dist / maxDist];
            }
        }

        // Convert hueSaturation and brightness to RGB
        let h = this.hueSaturation[0] + Math.PI;
        let s = this.hueSaturation[1];
        let v = this.brightness;

        // h *= 6;
        var i = Math.floor(h),
            f = h - i,
            p = v * (1 - s),
            q = v * (1 - f * s),
            t = v * (1 - (1 - f) * s),
            mod = i % 6;

        let r = [v, q, p, p, t, v][mod];
        let g = [t, v, v, q, p, p][mod];
        let b = [p, p, t, v, v, q][mod];

        this.value = [r,g,b, 1.0];
    }

    computeSize(width) {
        return [width, width - this.wY]
    }

    async serializeValue(nodeId, widgetIndex) {
        return [this.value[0], this.value[1], this.value[2], this.value[3]].toString();
    }
}

app.registerExtension({
    name: "jovimetrix.experimental",
    getCustomWidgets: () => {
        return {
            VEC4COLOR:  (node, inputName, inputData, app) => {
                return {
                    widget: node.addCustomWidget(new Vec4ColorWidget(inputName, inputData), ),
                    minWidth: 200,
                    minHeight: 300,
                }
            },
        }
    }
})