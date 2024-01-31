//
// Simple Checker Pattern
//

uniform vec2 uv_tile;

void main() {
    float result = mod(dot(vec2(1.0), step(vec2(0.5), fract(fragCoord * uv_tile))), 2.0);
    fragColor = mix(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), result);
}