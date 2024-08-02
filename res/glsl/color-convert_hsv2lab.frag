// name: HSV-2-LAB
// desc: Convert HSV input to LAB
// category: convert

uniform sampler2D image;

const float PI = 3.14159;
const vec3 D65 = vec3(95.047, 100.0, 108.883);

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);

    // Assuming input is in HSV format
    vec3 hsv = color.rgb;

    float H = hsv.x * 360.0;
    float S = hsv.y;
    float V = hsv.z;

    // Convert to LAB
    float L = V * 100.0;
    float C = S * L;

    float h = H * PI / 180.0;
    float a = C * cos(h);
    float b = C * sin(h);

    // Normalize LAB
    vec3 lab = vec3(L / 100.0, (a + 128.0) / 255.0, (b + 128.0) / 255.0);
    fragColor = vec4(lab, color.a);
}