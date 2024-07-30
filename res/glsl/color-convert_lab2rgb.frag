// name: LAB-2-RGB
// desc: Convert LAB color space to RGB
// category: convert

uniform sampler2D image;

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);

    // Denormalize LAB
    vec3 lab = color.rgb;
    float L = lab.x * 100.0;
    float a = lab.y * 255.0 - 128.0;
    float b = lab.z * 255.0 - 128.0;

    // LAB to XYZ
    float y = (L + 16.0) / 116.0;
    float x = a / 500.0 + y;
    float z = y - b / 200.0;

    vec3 xyz;
    xyz.x = 0.95047 * ((x * x * x > 0.008856) ? x * x * x : (x - 16.0 / 116.0) / 7.787);
    xyz.y = 1.00000 * ((y * y * y > 0.008856) ? y * y * y : (y - 16.0 / 116.0) / 7.787);
    xyz.z = 1.08883 * ((z * z * z > 0.008856) ? z * z * z : (z - 16.0 / 116.0) / 7.787);

    // XYZ to RGB
    vec3 rgb;
    rgb.r = xyz.x *  3.2406 + xyz.y * -1.5372 + xyz.z * -0.4986;
    rgb.g = xyz.x * -0.9689 + xyz.y *  1.8758 + xyz.z *  0.0415;
    rgb.b = xyz.x *  0.0557 + xyz.y * -0.2040 + xyz.z *  1.0570;

    // Apply gamma correction
    rgb = (rgb > 0.0031308) ? 1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055 : 12.92 * rgb;

    // Clamp to [0, 1] range
    rgb = clamp(rgb, 0.0, 1.0);

    fragColor = vec4(rgb, color.a);
}