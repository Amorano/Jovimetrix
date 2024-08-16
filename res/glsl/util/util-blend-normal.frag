// name: NORMAL BLEND
// desc: Blend two Normal maps
// category: UTILITY

uniform sampler2D imageA; //      | Input image A to blend with image B
uniform sampler2D imageB; //      | Input image B to blend with image A
uniform float blend;      // 0.50 | Intensity of blend

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = texture(imageA, uv);
    fragColor = vec4(col, 1.0);
}