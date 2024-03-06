float hash21(vec2 f){
    uvec2 p = floatBitsToUint(f);
    p = 1664525U*(p>>1U^p.yx);
    return float(1103515245U*(p.x^(p.y>>3U)))/float(0xffffffffU);
}

const vec2 s = vec2(1, 1.7320508);

float hex(in vec2 p){
    p = abs(p);
    return max(dot(p, s*.5), p.x); // Hexagon
}

vec4 getHex(vec2 p){
    vec4 hC = floor(vec4(p, p - vec2(.5, 1))/s.xyxy) + .5;
    vec4 h = vec4(p - hC.xy*s, p - (hC.zw + .5)*s);
    return dot(h.xy, h.xy)<dot(h.zw, h.zw) ? vec4(h.xy, hC.xy) : vec4(h.zw, hC.zw + .5);
}

void main() {
    vec2 u = fragCoord.xy; //(fragCoord - iResolution.xy*.5)/iResolution.y;
    vec4 h = getHex(u*5. + s.yx*iTime/6.);
    float eDist = hex(h.xy); // Edge distance.
    float cDist = dot(h.xy, h.xy); // Relative squared distance from the center.
    float rnd = hash21(h.zw);
    rnd = 1.;
    rnd = sin(rnd*6.283 + iTime*1.5)*.5 + .5;
    vec3 col = vec3(0, 0, 1);
    float blink = smoothstep(0., .125, rnd - .666); // Smooth blinking transition.
    float blend = dot(sin(u*6.283 - cos(u.yx*6.283)*3.14159), vec2(.25)) + .5; // Screen blend.
    col = max(col - mix(vec3(1., 0., 0.), vec3(0., 1., 1.), blend)*blink, 0.); // Blended, blinking orange.
    col = mix(col, col.xzy, dot(sin(u * 6. - cos(u * 1. + iTime)), vec2(.55)) + .4); // Orange and pink mix.
    if(h.y*.8660254<h.x*.5) col *= vec3(1.2, .35, .25);
    float cont = clamp(cos(eDist*6.283*12.) + .95, 0., 1.);
    cont = mix(cont, clamp(cos(eDist*6.283*12./2.)*1. + .95, 0., 1.), .125);
    col = mix(col, vec3(0), (1.-cont)*.95);
    col = mix(col, vec3(0), smoothstep(0., .03, eDist - .5 + .04));
    col *= max(1.25 - eDist*1.5, 0.);
    col *= max(1.25 - cDist*2., 0.);
    fragColor = vec4(sqrt(max(col, 0.)), 1);
}