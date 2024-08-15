// name: HSV-2-RGB
// desc: Convert HSV input to RGB
// category: COLOR/CONVERT

uniform sampler2D image; // | MASK, RGB or RGBA

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 c = texture(image, uv);

    // HSV to RGB conversion
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    fragColor = vec4(c.z * mix(vec3(1.0), rgb, c.y), color.a);
}