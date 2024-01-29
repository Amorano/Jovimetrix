//
// Remap to equal rectangle
//

#define PI  3.14159265359
#define TAU 6.28318530718

uniform bool flip;

void main() {
    vec2 uv = fragCoord * 2.0 - 1.0;
    float angle = (asin(uv.x) * acos(uv.x) + PI) / TAU;
    float radius = length(uv) * PI / TAU ;
    uv = vec2(angle, radius);
    // uv = normalize(uv);
    vec3 color;
    if (flip) {
        color = texture(iChannel0, uv).xyz;
    } else {
        color = texture(iChannel0, uv.yx).xyz;
    }
    fragColor = vec4(color, 1.0);
}