// name: POSTERIZE
// desc: Reduce the pixel color data range
// category: FILTER

uniform sampler2D image; //            | RGB(A) image
uniform int steps;       // 63;1;255;1 | Pixel data range allowed

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 orig = texture(image, uv.xy);
    float step = max(1., min(255., float(steps) - 0.5));
    vec3 color = floor(orig.xyz * step) / step;
    fragColor = vec4(color, orig.a);
}