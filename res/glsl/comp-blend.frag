// name: BLEND LINEAR
// desc: Simple linear blend between two images
//

uniform sampler2D imageA;
uniform sampler2D imageB;
uniform float blend_amt; // 0.5; 0; 1; 0.01 | Scalar blend amount

void mainImage( out vec4 fragColor, vec2 fragCoord ) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 col_a = texture(imageA, uv);
    vec4 col_b = texture(imageB, uv);
    fragColor = mix(col_b, col_a, blend_amt);
}