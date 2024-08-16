// name: HSV-2-RGB
// desc: Convert HSV image into RGB color space. Maintains alpha/mask.
// category: COLOR/CONVERT

uniform sampler2D image; // | HSV image

vec3 hsv2rgb(vec3 c) {
    c = vec3(c.x, clamp(c.yz, 0.0, 1.0));
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 c = texture(image, uv);
    fragColor = vec4(hsv2rgb(color.rgb), color.a);
}