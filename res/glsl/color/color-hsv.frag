// name: HSV ADJUST
// desc: Hue, Saturation and Value adjustment control. Maintains alpha/mask.
// category: COLOR

uniform sampler2D image; //                      | RGB(A) image
uniform vec3 HSV;        // 0.,1.,1.;-1;2;0.01 | Adjust the Hue, Saturation or Value

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + M_EPSILON)), d / (q.x + M_EPSILON), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);
    vec3 rgb = color.rgb;
    vec3 hsv = rgb2hsv(rgb);

    // Adjust hue
    hsv.x = mod(hsv.x + HSV.x, 1.0);

    // Adjust saturation
    hsv.y = clamp(hsv.y * HSV.y, 0.0, 1.0);

    // Adjust value
    hsv.z = clamp(hsv.z * HSV.z, 0.0, 1.0);

    rgb = hsv2rgb(hsv);
    fragColor = vec4(rgb, color.a);
}