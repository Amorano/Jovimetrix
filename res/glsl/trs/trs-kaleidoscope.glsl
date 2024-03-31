//  shader-kaleidoscope
// https://github.com/Phlox-GL/shader-kaleidoscope/blob/main/shaders/kaleidoscope.frag

const float TAU = 6.28318530718;

// divide circle by X segments
uniform float segments;
// radius of the circle containing the shape
uniform float radius;
uniform float regress;
// offset through the texture per segment
uniform vec2 shift;
// spin of background image
uniform float spin;
uniform float scale;
uniform float skip;

float rand(vec2 co) {
    return fract(sin(dot(co * 1000.0, vec2(12.9898, 78.233))) * 43758.5453) / 10.;
}

float square(float x) {
    return x * x;
}

vec2 product(vec2 a, vec2 b) {
    return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

vec2 conjugate(vec2 a) {
    return vec2(a.x,-a.y);
}

vec2 divide(vec2 a, vec2 b) {
    return vec2(((a.x*b.x+a.y*b.y)/(b.x*b.x+b.y*b.y)),((a.y*b.x-a.x*b.y)/(b.x*b.x+b.y*b.y)));
}

vec2 perpendicular(vec2 p, vec2 p1, vec2 p2) {
    float x = p.x;
    float y = p.y;
    float a = p1.x;
    float b = p1.y;
    float c = p2.x;
    float d = p2.y;
    // corrected with https://blog.csdn.net/qq_32867925/article/details/114294753
    float k = - ((a-x)*(c-a) + (b-y)*(d-b)) / (square(a-c) + square(b-d));
    return vec2(
        a + (c - a) * k,
        b + (d - b) * k
    );
}

bool is_outside_line(vec2 p, vec2 p1, vec2 p2) {
    vec2 perp = perpendicular(p, p1, p2);
    float l = length(perp);
    return product(p, conjugate(perp)).x / (l*l) > 1.0;
}

vec2 rotate_by_radian(vec2 p, float a) {
    vec2 rot = vec2(cos(a), sin(a));
    return product(p, rot);
}

vec2 reflection_line(vec2 p, vec2 p1, vec2 p2) {
    vec2 perp = perpendicular(p, p1, p2);
    vec2 d = perp - p;
    float ld = length(d);
    return perp + (d+ (-skip * d / ld)) * regress;
}

void main() {
    vec4 bgColor = vec4(0.1, 0.2, 0.4, 1.0);
    vec4 moonColor = vec4(1.0, 1.0, 0.5, 1.0);
    vec4 spotColor = vec4(0.9, 0.9, 0.5, 1.0);
    float unit = TAU / segments;
    vec2 color_point = vec2(fragCoord.x + 0.5, 0.5 - fragCoord.y);
    for (int i = 0; i < 40; ++i) {
        float point_angle = atan(color_point.y, color_point.x);
        float at_part = floor(point_angle / unit);
        vec2 p1 = vec2(cos(at_part * unit), sin(at_part * unit)) * radius;
        vec2 p2 = vec2(cos((at_part + 1.0) * unit), sin((at_part + 1.0) * unit)) * radius;
        vec2 perp = perpendicular(color_point, p1, p2);
        if (is_outside_line(color_point, p1, p2)) {
            color_point = reflection_line(color_point, p1, p2);
            continue;
        } else {
            vec2 spin_rot = vec2(cos(spin), sin(spin));
            color_point = product((color_point / scale), spin_rot);
            fragColor = texture2D(iChannel0, fract(color_point - shift));
            return;
        }
        return;
    }
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}