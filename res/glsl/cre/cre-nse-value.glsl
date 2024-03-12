// Value noise by Inigo Quilez - iq/2013
// https://www.shadertoy.com/view/lsf3WH

uniform int seed;
uniform vec2 uTile;

float random (in vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233)) * seed) * 43758.5453123);
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix( mix( random( i + vec2(0.0,0.0) ),
                     random( i + vec2(1.0,0.0) ), u.x),
                mix( random( i + vec2(0.0,1.0) ),
                     random( i + vec2(1.0,1.0) ), u.x), u.y);
}

float lines(in vec2 pos, float b){
    float scale = 10.0;
    pos *= scale;
    return smoothstep(0.0,
                    .5 + b * .5,
                    abs((sin(pos.x * 3.1415)+ b * 2.0)) * .5);
}

void main() {
    vec2 pos = fragCoord.yx * uTile.yx;
    float pattern = pos.x;
    pos = noise(pos) * pos;
    pattern = lines(pos, .5);
    fragColor = vec4(vec3(pattern), 1.0);
}
