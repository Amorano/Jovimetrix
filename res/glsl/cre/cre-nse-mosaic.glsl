//
// Mosaic Noise
//

uniform float scalar; // 10.

float random (vec2 pos) {
    return fract(sin(dot(pos.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
    vec2 uv = fragCoord * scalar;
    vec2 i = floor(uv);
    // vec2 f = fract(uv);
    vec3 color = vec3(random(i));
    fragColor = vec4(color, 1.0);
}
