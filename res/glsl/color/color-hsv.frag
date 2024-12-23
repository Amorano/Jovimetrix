// name: HSV ADJUST
// desc: Hue, Saturation and Value adjustment control. Maintains alpha/mask.
// category: COLOR

#include .lib/convert.lib

uniform sampler2D image; //                    | RGB(A) image
uniform vec3 HSV;        // 0.,1.,1.;-1;2;0.01 | Adjust the Hue, Saturation or Value

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);
    vec3 hsv = convert_rgb2hsv(color.rgb);

    hsv.x = mod(hsv.x + HSV.x, 1.0);
    hsv.y = clamp(hsv.y * HSV.y, 0.0, 1.0);
    hsv.z = clamp(hsv.z * HSV.z, 0.0, 1.0);
    fragColor = vec4(convert_hsv2rgb(hsv), color.a);
}