// JOVIMETRIX GL SHADER
// Brownian Noise
// MIT License

uniform float radius; // 2
uniform float strength; // 1.
uniform vec2 center; // 0.5, 0.5

vec2 bulge(vec2 uv, vec2 pivot) {
    uv -= pivot;
    float dist = length(uv) / (1. / radius) / 8.;
    float str = strength / (1.0 + pow(dist, 2.));
    uv *= str;
    uv += pivot;
    return uv;
}

void main() {
    vec2 uv = bulge(iUV, center);
    vec4 tex = texture(iChannel0, uv);
    fragColor = vec4(tex.rgb, 1.0);
}