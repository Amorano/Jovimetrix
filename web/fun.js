/**/

import { app } from "../../scripts/app.js";

export const bewm = function(ex, ey) {
    //- adapted from "Anchor Click Canvas Animation" by Nick Sheffield
    //- https://codepen.io/nicksheffield/pen/NNEoLg/
    const colors = [ '#ffc000', '#ff3b3b', '#ff8400' ];
    const bubbles = 25;

    const explode = () => {
        let particles = [];
        const ctx = app.canvas;
        const canvas = ctx.getContext('2d');
        ctx.style.pointerEvents = 'none';

        for(var i = 0; i < bubbles; i++) {
            particles.push({
                x: canvas.width / 2,
                y: canvas.height / 2,
                radius: r(20, 30),
                color: colors[Math.floor(Math.random() * colors.length)],
                rotation: r(0, 360, true),
                speed: r(12, 16),
                friction: 0.9,
                opacity: r(0, 0.5, true),
                yVel: 0,
                gravity: 0.15
            });
        }
        render(particles, ctx);
    }

    const render = (particles, ctx) => {
        requestAnimationFrame(() => render(particles, ctx));

        particles.forEach((p) => {
            p.x += p.speed * Math.cos(p.rotation * Math.PI / 180);
            p.y += p.speed * Math.sin(p.rotation * Math.PI / 180);

            p.opacity -= 0.01;
            p.speed *= p.friction;
            p.radius *= p.friction;
            p.yVel += p.gravity;
            p.y += p.yVel;

            if(p.opacity < 0 || p.radius < 0) return;

            ctx.beginPath();
            ctx.globalAlpha = p.opacity;
            ctx.fillStyle = p.color;
            ctx.arc(p.x, p.y, p.radius, 0, 2 * Math.PI, false);
            ctx.fill();
        });
    }

    const r = (a, b, c) => parseFloat((Math.random() * ((a ? a : 1) - (b ? b : 0)) + (b ? b : 0)).toFixed(c ? c : 0));
    explode(ex, ey);
}

export const bubbles = function() {
    const canvas = document.getElementById("graph-canvas");
    const context = canvas.getContext("2d");
    window.bubbles_alive = true;
    let mouseX;
    let mouseY;

    const particleArray = [];
    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width * 0.85;
            this.y = canvas.height * 0.85;
            this.radius = Math.random() * 30;
            this.dx = Math.random() - 0.5
            this.dx = Math.sign(this.dx) * Math.random() * 1.27;
            this.dy = 3 + Math.random() * 3;
            this.hue = 25 + Math.random() * 250;
            this.sat = 85 + Math.random() * 15;
            this.val = 35 + Math.random() * 20;
        }

        draw() {
            context.beginPath();
            context.arc(this.x, this.y, this.radius, 0, 2 * Math.PI);
            context.strokeStyle = `hsl(${this.hue} ${this.sat}% ${this.val}%)`;
            context.stroke();

            const gradient = context.createRadialGradient(
                this.x,
                this.y,
                1,
                this.x + 0.5,
                this.y + 0.5,
                this.radius
            );

            gradient.addColorStop(0.3, "rgba(255, 255, 255, 0.3)");
            gradient.addColorStop(0.95, "#E7FEFF7F");
            context.fillStyle = gradient;
            context.fill();
        }

        move() {
            this.x = this.x + this.dx + (Math.random() - 0.5) * 0.5;
            this.y = this.y - this.dy + (Math.random() - 0.5) * 1.5;

            // Check if the particle is outside the canvas boundaries
            if (
                this.x < -this.radius ||
                this.x > canvas.width + this.radius ||
                this.y < -this.radius ||
                this.y > canvas.height + this.radius
            ) {
                // Remove the particle from the array
                particleArray.splice(particleArray.indexOf(this), 1);
            }
        }
    }

    const animate = () => {
        //context.clearRect(0, 0, canvas.width, canvas.height);
        app.canvas.setDirty(true);

        particleArray.forEach((particle) => {
            particle?.move();
            particle?.draw();
        });

        if (window.bubbles_alive) {
            requestAnimationFrame(animate);
            if (Math.random() > 0.5) {
                const particle = new Particle(mouseX, mouseY);
                particleArray.push(particle);
            }
        } else {
            canvas.removeEventListener("mousemove", handleMouseMove);
            particleArray.length = 0; // Clear the particleArray
            return;
        }
    };

    const handleMouseMove = (event) => {
        mouseX = event.clientX;
        mouseY = event.clientY;
    };

    canvas.addEventListener("mousemove", handleMouseMove);
    animate();
}

// flash status for each element
const flashStatusMap = new Map();

export async function flashBackgroundColor(element, duration, flashCount, color="red") {
    if (flashStatusMap.get(element)) {
        return;
    }

    flashStatusMap.set(element, true);
    const originalColor = element.style.backgroundColor;

    for (let i = 0; i < flashCount; i++) {
        element.style.backgroundColor = color;
        await new Promise(resolve => setTimeout(resolve, duration / 2));
        element.style.backgroundColor = originalColor;
        await new Promise(resolve => setTimeout(resolve, duration / 2));
    }
    flashStatusMap.set(element, false);
}