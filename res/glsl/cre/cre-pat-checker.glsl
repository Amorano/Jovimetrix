//
// Simple Checker Pattern
//

uniform vec2 uTile;

void main() {
    vec2 st = fract(fragCoord * uTile);
    float result = mod(dot(vec2(1.0), step(vec2(0.5), st)), 2.0);
    fragColor = mix(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), result);
}