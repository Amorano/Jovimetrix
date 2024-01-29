//
// Tile inputs
//

uniform float uTime;
uniform vec2 uTile;

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647693

vec2 rotate2D(vec2 _st, float _angle) {
    _st -= 0.5;
    _st =  mat2(cos(_angle),-sin(_angle),
                sin(_angle),cos(_angle)) * _st;
    _st += 0.5;
    return _st;
}

vec2 tile(vec2 _st, vec2 _zoom) {
    _st.x *= _zoom.x;
    _st.y *= _zoom.y;
    return fract(_st);
}

vec2 tile_rotate(vec2 _st){
    _st *= 2.0;

    float index = 0.0;
    if (fract(_st.x * 0.5) > 0.5){
        index += 1.0;
    }
    if (fract(_st.y * 0.5) > 0.5){
        index += 2.0;
    }

    _st = fract(_st);

    if(index == 1.0){
        _st = rotate2D(_st,PI*0.5);
    } else if(index == 2.0){
        _st = rotate2D(_st,PI*-0.5);
    } else if(index == 3.0){
        _st = rotate2D(_st,PI);
    }

    return _st;
}

vec2 tile_mirror(vec2 _st, float _zoom){
    _st *= _zoom;
    if (fract(_st.y * 0.5) > 0.5){
        _st.x = _st.x+0.5;
        _st.y = 1.0-_st.y;
    }
    return fract(_st);
}

float pattern_sawtooth(vec2 st, float _pct, float blend) {
    st = tile_mirror(st * vec2(1., 2.), 5.);
    float x = st.x * 2.;
    float a = floor(1. + sin(x * PI));
    float b = floor(1. + sin((x + 1.) * PI));
    float f = fract(x);
    return smoothstep( _pct - blend, _pct, st.y);
}

// Based on https://www.shadertoy.com/view/4sSSzG
float pattern_triangle (vec2 _st,
                vec2 _p0, vec2 _p1, vec2 _p2,
                float _smoothness) {

    vec3 e0, e1, e2;

    e0.xy = normalize(_p1 - _p0).yx * vec2(+1.0, -1.0);
    e1.xy = normalize(_p2 - _p1).yx * vec2(+1.0, -1.0);
    e2.xy = normalize(_p0 - _p2).yx * vec2(+1.0, -1.0);

    e0.z = dot(e0.xy, _p0) - _smoothness;
    e1.z = dot(e1.xy, _p1) - _smoothness;
    e2.z = dot(e2.xy, _p2) - _smoothness;

    float a = max(0.0, dot(e0.xy, _st) - e0.z);
    float b = max(0.0, dot(e1.xy, _st) - e1.z);
    float c = max(0.0, dot(e2.xy, _st) - e2.z);

    return smoothstep(_smoothness * 2.0,
                    1e-7,
                    length(vec3(a, b, c)));
}

void main (void) {

    vec2 st = tile(fragCoord, uTile);
    //st = tile_rotate(st);
    st = rotate2D(st, -PI * uTime * 0.25);
    vec4 color = texture(iChannel0, st);

    float pattern = pattern_triangle(st,
                        vec2(0.30, - 0.5),
                        vec2(0.70, 0. - 0.5),
                        vec2(0.5, 1.0),
                        0.01);

    // float pattern = pattern_sawtooth(st, mix(a, b, f), 0.01)
    fragColor = vec4(color.xyz, 1.0);
}