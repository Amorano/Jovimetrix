// name: TRANSFORM
// desc: Abuse UV space to create repetitions. Maximums are row: width/4; column: height/4
// category: MODIFY

#include .lib/const.lib

uniform sampler2D image; //                        | RGB(A) input to repeat
uniform vec2 offset;     // 0.0,0.0;-0.5;0.5;0.001 | positional offset (-0.5..0.5)
uniform float rotate;    // 0;0;1;0.001            | rotation from 0..2pi
uniform vec2 tile;       // 1.0,1.0;1;2048;1       | repetitions on X and Y

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // normalize + offset
    vec2 uv = (fragCoord - offset * iResolution.xy) / iResolution.xy;

    // rotation matrix
    float cosAngle = cos(rotate * M_TAU);
    float sinAngle = sin(rotate * M_TAU);
    mat2 rotationMatrix = mat2(cosAngle, -sinAngle, sinAngle, cosAngle);

	// center rotate, scale
    uv = rotationMatrix * (uv - 0.5) + 0.5;
	vec2 repeat = vec2(min(iResolution.x / 4., tile.x), min(iResolution.y / 4., tile.y));
    uv *= repeat;
    fragColor = texture(image, uv);
}