// name: LAB-2-RGB
// desc: Convert LAB input to RGB
// category: COLOR/CONVERT

uniform sampler2D image; // | MASK, RGB or RGBA

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);

    // Denormalize LAB from 0-1 range
    vec3 lab;
    lab.x = color.r * 100.0;  // L: 0 to 1 -> 0 to 100
    lab.y = color.g * 255.0 - 128.0;  // a: 0 to 1 -> -128 to 127
    lab.z = color.b * 255.0 - 128.0;  // b: 0 to 1 -> -128 to 127

    // LAB to XYZ
    vec3 f;
    f.y = (lab.x + 16.0) / 116.0;
    f.x = lab.y / 500.0 + f.y;
    f.z = f.y - lab.z / 200.0;

    vec3 xyz;
    xyz.x = (f.x > 0.206897) ? pow(f.x, 3.0) : (f.x - 16.0 / 116.0) / 7.787;
    xyz.y = (f.y > 0.206897) ? pow(f.y, 3.0) : (f.y - 16.0 / 116.0) / 7.787;
    xyz.z = (f.z > 0.206897) ? pow(f.z, 3.0) : (f.z - 16.0 / 116.0) / 7.787;

    // xyz *= vec3(109.85, 100.00, 35.58);  // CIE A -> 109.85, 100.00, 35.58
    // xyz *= vec3(96.42, 100.00, 82.51);  // D50 -> 96.42, 100.00, 82.51
    // xyz *= vec3(95.68, 100.00, 92.14);  // D55 -> 95.68, 100.00, 92.14
    xyz *= vec3(0.95047, 1.0, 1.08883);  // D65 -> 95.047, 100.0, 108.883
    // xyz *= vec3(96.42, 100.0, 82.49);  // ICC -> 96.42, 100.0, 82.49

    // XYZ to RGB
    vec3 rgb = xyz / 100.0 * mat3(
         3.2404542, -1.5371385, -0.4985314,
        -0.9692660,  1.8760108,  0.0415560,
         0.0556434, -0.2040259,  1.0572252
    );

    // Apply gamma correction
    rgb = vec3(
        (rgb.r > 0.0031308) ? 1.055 * pow(rgb.r, 1.0 / 2.4) - 0.055 : 12.92 * rgb.r,
        (rgb.g > 0.0031308) ? 1.055 * pow(rgb.g, 1.0 / 2.4) - 0.055 : 12.92 * rgb.g,
        (rgb.b > 0.0031308) ? 1.055 * pow(rgb.b, 1.0 / 2.4) - 0.055 : 12.92 * rgb.b
    );

    // Scale back to 0-255 range
    rgb = clamp(rgb * 255.0, 0.0, 255.0);
    fragColor = vec4(rgb, color.a);
}