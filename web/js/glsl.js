// Create a canvas element
const canvas = document.createElement('canvas');
canvas.width = 512;
canvas.height = 512;
document.body.appendChild(canvas);

// Try to obtain a WebGL context
const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

if (!gl) {
  console.log('Unable to initialize WebGL. Your browser may not support it.');
} else {
  console.log('WebGL context initialized successfully.');
}