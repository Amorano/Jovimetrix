// name: POSTERIZE
// desc: Reduce the pixel color data range
// category: COLOR

#include .lib/color.lib

uniform sampler2D image; //            | RGB(A) image
uniform int steps;       // 16;2;255;1 | Pixel data range allowed

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 orig = texture(image, uv.xy);
    vec3 color = color_posterize(orig.rgb, steps);
    fragColor = vec4(color, orig.a);
}