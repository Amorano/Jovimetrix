//
// Mirror Input
//

uniform float uZoom;
uniform vec2 center;

void main(){
  vec2 st = fragCoord;
  if (fract(st.x * 0.5) > center.x) {
    st.x = 1.0 - st.x;
  }
  if (fract(st.y * 0.5) > center.y) {
    st.y = 1.0 - st.y;
  }
  st = fract(st) * uZoom;
  fragColor = texture(iChannel0, st);
}