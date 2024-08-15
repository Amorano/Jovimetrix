// name: RGB-2-LAB
// desc: Convert RGB input to LAB
// category: COLOR/CONVERT

uniform sampler2D image; // | MASK, RGB or RGBA

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);

    // RGB to XYZ
    vec3 rgb = color.rgb / 255.0;
    vec3 xyz;
    xyz.r = (rgb.r > 0.04045) ? pow((rgb.r + 0.055) / 1.055, 2.4) : rgb.r / 12.92;
    xyz.g = (rgb.g > 0.04045) ? pow((rgb.g + 0.055) / 1.055, 2.4) : rgb.g / 12.92;
    xyz.b = (rgb.b > 0.04045) ? pow((rgb.b + 0.055) / 1.055, 2.4) : rgb.b / 12.92;

    xyz = xyz * mat3(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    );

    // XYZ to LAB
    // xyz /= vec3(109.85, 100.00, 35.58);  // CIE A -> 109.85, 100.00, 35.58
    // xyz /= vec3(96.42, 100.00, 82.51);  // D50 -> 96.42, 100.00, 82.51
    // xyz /= vec3(95.68, 100.00, 92.14);  // D55 -> 95.68, 100.00, 92.14
    xyz /= vec3(0.95047, 1.0, 1.08883);  // D65 -> 95.047, 100.0, 108.883
    // xyz /= vec3(96.42, 100.0, 82.49);  // ICC -> 96.42, 100.0, 82.49

    vec3 f;
    f.x = (xyz.x > 0.008856) ? pow(xyz.x, 1.0/3.0) : (7.787 * xyz.x) + 16.0/116.0;
    f.y = (xyz.y > 0.008856) ? pow(xyz.y, 1.0/3.0) : (7.787 * xyz.y) + 16.0/116.0;
    f.z = (xyz.z > 0.008856) ? pow(xyz.z, 1.0/3.0) : (7.787 * xyz.z) + 16.0/116.0;

    vec3 lab;
    lab.x = (116.0 * f.y) - 16.0;    // L
    lab.y = 500.0 * (f.x - f.y);     // a
    lab.z = 200.0 * (f.y - f.z);     // b

    // Normalize LAB to 0-1 range for output
    lab.x = lab.x / 100.0;  // L: 0 to 100 -> 0 to 1
    lab.y = (lab.y + 128.0) / 255.0;  // a: -128 to 127 -> 0 to 1
    lab.z = (lab.z + 128.0) / 255.0;  // b: -128 to 127 -> 0 to 1

    fragColor = vec4(lab, color.a);
}
