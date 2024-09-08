// name: CONICAL GRADIENT
// desc: Generate a conical gradient from black to white
// category: CREATE

uniform vec2 origin;       // 0.5,0.5;0;1;0.01 | Intensity of base normal
uniform vec2 range;        // 0.0,1.0;0;1;0.01 | start of range. 0=start. size of range. 1=full range.
uniform float angleOffset; // 0.0; 0; 1; 0.01  | offset of the gradient starting angle

#define TAU 6.283185307179586

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    uv -= origin;
    float angle = atan(uv.y, uv.x) + angleOffset * -TAU;
    float norm = mod(angle, TAU);
    norm = (norm / TAU + range.x) * range.y;
    fragColor = vec4(vec3(norm), 1.0);
}