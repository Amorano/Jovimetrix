// name: LAB-2-HSV
// desc: Convert LAB color space to HSV
// category: COLOR/CONVERT

uniform sampler2D image; // | MASK, RGB or RGBA

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);

    // De-normalize LAB
    vec3 lab = color.rgb;
    lab.x *= 100.0;
    lab.y = lab.y * 255.0 - 128.0;
    lab.z = lab.z * 255.0 - 128.0;

    // Calculate chroma and hue
    float C = sqrt(lab.y * lab.y + lab.z * lab.z);
    float h = atan(lab.z, lab.y);

    // Convert to HSV
    float H = h * 57.29577951; // Convert radians to degrees (180 / pi)
    if (H < 0.0) H += 360.0;

    float S = C / (lab.x + 1e-5); // Avoid division by zero
    float V = lab.x / 100.0;

    // Normalize H to 0-1 range
    H /= 360.0;

    fragColor = vec4(H, S, V, color.a);
}