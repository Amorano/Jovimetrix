// name: HSV-2-RGB
// desc: Convert HSV input to RGB
// category: COLOR/CONVERT

uniform sampler2D image;

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);
    vec3 hsv = color.rgb;  // Assuming input is in HSV format

    // HSV to RGB conversion
    vec3 rgb;
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(hsv.xxx + K.xyz) * 6.0 - K.www);
    rgb = hsv.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);

    fragColor = vec4(rgb, color.a);
}