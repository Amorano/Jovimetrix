//

uniform int sides;
uniform float radius;

#define PI 3.14159265359
#define TAU 6.28318530718

void main(){
  vec2 uv = iUV;
  uv.x *= iResolution.x / iResolution.y;
  uv = uv * 2. - 1.;

  float a = atan(uv.x, uv.y) + PI;
  float r = TAU / float(sides);
  float d = cos(floor(0.5 + a / r) * r - a) * length(uv) * radius;
  vec3 color = vec3(1.0 - smoothstep(.4, .41, d));
  fragColor = vec4(color, 1.0);
}