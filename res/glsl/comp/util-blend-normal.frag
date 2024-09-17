// name: NORMAL BLEND
// desc: Blend two Normal maps
// category: COMPOSE

uniform sampler2D imageA; //                 | Input image A to blend with image B
uniform sampler2D imageB; //                 | Input image B to blend with image A
uniform float blend;      // 0.5; 0; 1; 0.01 | Intensity of blend

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;

    vec3 normalA = texture(imageA, uv).rgb * 2.0 - 1.0;
    normalA = normalize(normalA);

    vec3 normalB = texture(imageB, uv).rgb * 2.0 - 1.0;
    normalB = normalize(normalB);

    vec3 blendedNormal = normalize(mix(normalA, normalB, blend));
    fragColor = vec4((blendedNormal * 0.5) + 0.5, 1.0);
}