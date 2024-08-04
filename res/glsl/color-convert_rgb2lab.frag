// name: RGB-2-LAB
// desc: Convert RGB input to LAB
// category: convert

uniform sampler2D image;

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);

    // RGB to XYZ
    vec3 rgb = color.rgb / 255;
    vec3 xyz;
    xyz.r = (rgb.r > 0.04045) ? pow((rgb.r + 0.055) / 1.055, 2.4) : rgb.r / 12.92;
    xyz.g = (rgb.g > 0.04045) ? pow((rgb.g + 0.055) / 1.055, 2.4) : rgb.g / 12.92;
    xyz.b = (rgb.b > 0.04045) ? pow((rgb.b + 0.055) / 1.055, 2.4) : rgb.b / 12.92;
    xyz *= 100;

    xyz = xyz * mat3(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    );

    // XYZ to LAB
    xyz /= vec3(95.047, 100.0, 108.883); // D65 reference white

    vec3 f;
    f.x = (xyz.x > 0.008856) ? pow(xyz.x, 0.33333333) : (7.787 * xyz.x) + 0.13793103; // 16.0 / 116.0 = 0.13793103
    f.y = (xyz.y > 0.008856) ? pow(xyz.y, 0.33333333) : (7.787 * xyz.y) + 0.13793103;
    f.z = (xyz.z > 0.008856) ? pow(xyz.z, 0.33333333) : (7.787 * xyz.z) + 0.13793103;

    vec3 lab;
    lab.x = (116.0 * f.y) - 16.0;    // L
    lab.y = 500.0 * (f.x - f.y);     // a
    lab.z = 200.0 * (f.y - f.z);     // b

    fragColor = vec4(lab, color.a);
}
