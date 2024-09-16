// name: CONICAL GRADIENT
// desc: Generate a conical gradient from black to white
// category: CREATE

#include .lib/const.lib

uniform vec2 origin;       // 0.5,0.5;0;1;0.01 | Intensity of base normal
uniform vec2 range;        // 0.0,1.0;0;1;0.01 | start of range. 0=start. size of range. 1=full range.
uniform float angleOffset; // 0.0; 0; 1; 0.01  | offset of the gradient starting angle

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    uv -= origin;
    float angle = atan(uv.y, uv.x) + angleOffset * -M_TAU;
    float norm = mod(angle, M_TAU);
    norm = (norm / M_TAU + range.x) * range.y;
    fragColor = vec4(vec3(norm), 1.0);
}