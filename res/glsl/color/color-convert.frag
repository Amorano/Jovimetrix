// name: COLOR CONVERSION
// desc: Convert an image from one color space (RGB, HSV, LAB, XYZ) to another.
// category: COLOR

#include .lib/color.lib

uniform sampler2D image; // | Image to convert
uniform int operator;    // EnumGLSLColorConvert | conversion operation to perform.

// =============================================================================
// SELECTOR
// =============================================================================

vec3 convertColor(vec3 color, int operator) {
    // RGB
    if (operator == 0) {
        return rgb2hsv(color);
    } else if (operator == 1) {
        return rgb2lab(color);
    } else if (operator == 2) {
        return rgb2xyz(color);
    // HSV
    } else if (operator == 10) {
        return hsv2rgb(color);
    } else if (operator == 11) {
        return hsv2lab(color);
    } else if (operator == 12) {
        return hsv2xyz(color);
    // LAB
    } else if (operator == 20) {
        return lab2rgb(color);
    } else if (operator == 21) {
        return lab2hsv(color);
    } else if (operator == 22) {
        return lab2xyz(color);
    // XYZ
    } else if (operator == 30) {
        return xyz2rgb(color);
    } else if (operator == 31) {
        return xyz2hsv(color);
    } else if (operator == 32) {
        return xyz2lab(color);
    }
    return color;
}

void mainImage(out vec4 fragColor, vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 color = texture(image, uv);
    vec3 rgb = convertColor(color.rgb, operator);
    fragColor = vec4(rgb, color.a);
}