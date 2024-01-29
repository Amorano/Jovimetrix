// Feel free to steal this :^)
// Consider it MIT licensed, you can link to this page if you want to.
// https://www.shadertoy.com/view/4t2fRz

// iUser1 == INTENSITY
// iUser2 == SPEED

#define SHOW_NOISE 0
#define SRGB 0
// 0: Addition, 1: Screen, 2: Overlay, 3: Soft Light, 4: Lighten-Only
#define BLEND_MODE 0
// What gray level noise should tend to.
#define MEAN 0.0
// Controls the contrast/variance of noise.
#define VARIANCE 0.5

vec3 channel_mix(vec3 a, vec3 b, vec3 w) {
    return vec3(mix(a.r, b.r, w.r), mix(a.g, b.g, w.g), mix(a.b, b.b, w.b));
}

float gaussian(float z, float u, float o) {
    return (1.0 / (o * sqrt(2.0 * 3.1415))) * exp(-(((z - u) * (z - u)) / (2.0 * (o * o))));
}

vec3 madd(vec3 a, vec3 b, float w) {
    return a + a * b * w;
}

vec3 screen(vec3 a, vec3 b, float w) {
    return mix(a, vec3(1.0) - (vec3(1.0) - a) * (vec3(1.0) - b), w);
}

vec3 overlay(vec3 a, vec3 b, float w) {
    return mix(a, channel_mix(
        2.0 * a * b,
        vec3(1.0) - 2.0 * (vec3(1.0) - a) * (vec3(1.0) - b),
        step(vec3(0.5), a)
    ), w);
}

vec3 soft_light(vec3 a, vec3 b, float w) {
    return mix(a, pow(a, pow(vec3(2.0), 2.0 * (vec3(0.5) - b))), w);
}

void main() {
    //vec2 ps = vec2(1.0) / iResolution.xy;
    //vec2 uv = fragCoord * ps;
    fragColor = texture(iChannel0, fragCoord);
    #if SRGB
        fragColor = pow(fragColor, vec4(2.2));
    #endif

    float t = iTime * float(iUser2);
    float seed = dot(fragCoord, vec2(12.9898, 78.233));
    float noise = fract(sin(seed) * 43758.5453 + t);
    noise = gaussian(noise, float(MEAN), float(VARIANCE) * float(VARIANCE));

    #if SHOW_NOISE
        fragColor = vec4(noise);
    #else
        float w = float(iUser1);
        vec3 grain = vec3(noise) * (1.0 - fragColor.rgb);

        #if BLEND_MODE == 0
            fragColor.rgb += grain * w;
        #elif BLEND_MODE == 1
            fragColor.rgb = screen(fragColor.rgb, grain, w);
        #elif BLEND_MODE == 2
            fragColor.rgb = overlay(fragColor.rgb, grain, w);
        #elif BLEND_MODE == 3
            fragColor.rgb = soft_light(fragColor.rgb, grain, w);
        #elif BLEND_MODE == 4
            fragColor.rgb = max(fragColor.rgb, grain * w);
        #endif

        #if SRGB
            fragColor = pow(fragColor, vec4(1.0 / 2.2));
        #endif
    #endif
}
