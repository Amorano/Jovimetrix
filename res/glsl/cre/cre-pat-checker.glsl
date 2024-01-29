//
// Simple Checker Pattern
//

uniform float iUser3;

void main() {
    float result = mod(dot(vec2(1.0), step(vec2(0.5), fract(fragCoord * iUser3))), 2.0);
    fragColor = mix(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), result);
}