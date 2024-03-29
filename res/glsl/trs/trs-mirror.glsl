#define PI 3.14159265358979323846
#define TAU PI * 2.0

uniform vec2 u_resolution;
uniform float u_time;

float fill(float _st, float _pct, float _antia){
  return smoothstep( _pct - _antia, _pct, _st);
}

vec2 mirrorTile(vec2 _st, float _zoom){
    _st *= _zoom;
    if (fract(_st.y * 0.5) > 0.5){
        _st.y = 1.0 - _st.y;
    }
    return fract(_st);
}

void main(){
  vec2 st = mirrorTile(fragCoord, 1.0);
  vec3 color = vec3(fill(st.y, 0.5 + sin(st.x * TAU) * 0.45, 0.02));
  gl_FragColor = vec4( color, 1.0 );
}