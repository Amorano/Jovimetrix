// name: LAB-2-RGB
// desc: Convert LAB input to RGB
// category: convert

uniform sampler2D image;

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);
    vec3 lab = color.rgb;

    // LAB to XYZ
    vec3 f;
    f.y = (lab.x + 16.0) / 116.0;
    f.x = lab.y / 500.0 + f.y;
    f.z = f.y - lab.z / 200.0;

    vec3 xyz;
    xyz.x = (f.x > 0.206897) ? pow(f.x, 3.0) : (f.x - 16.0 / 116.0) / 7.787;
    xyz.y = (f.y > 0.206897) ? pow(f.y, 3.0) : (f.y - 16.0 / 116.0) / 7.787;
    xyz.z = (f.z > 0.206897) ? pow(f.z, 3.0) : (f.z - 16.0 / 116.0) / 7.787;

    // D65 reference white
    xyz *= vec3(95.047, 100.0, 108.883);

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
    rgb *= 255.0;

    fragColor = vec4(rgb, color.a);
}