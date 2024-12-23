// name: INVERT
// desc: Invert the channels of an image along a scalar [0..1] range.
// category: COLOR

uniform sampler2D image;  //             | 4-channel data
uniform vec4 invert;      // 0,0,0,0;0;1 | amount to invert each channel

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 col = texture(image, uv);
	fragColor = mix(col, 1.0 - col, invert);
}