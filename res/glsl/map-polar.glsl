// JOVIMETRIX GL SHADER
// Convert input to grayscale
// MIT License

#define PI  3.14159265359
#define TAU 6.28318530718

uniform bool flip;

void main() {
    vec2 uv = iUV * 2.0 - 1.0;
    float angle = (atan(uv.y, uv.x) + PI) / TAU;
    float radius = length(uv) / sqrt(2.0);
    uv = vec2(angle, radius);
    vec3 color;
    if (flip) {
        color = texture(iChannel0, uv).xyz;
    } else {
        color = texture(iChannel0, uv.yx).xyz;
    }
    fragColor = vec4(color, 1.0);
}