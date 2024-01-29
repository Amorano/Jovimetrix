uniform sampler2D tex0;
uniform float time;
uniform float rt_w;
uniform float rt_h;
uniform float stitching_size = 6.0;
uniform int invert = 0;

vec4 PostFX(sampler2D tex, vec2 uv, float time)
{
  vec4 c = vec4(0.0);
  float size = stitching_size;
  vec2 cPos = uv * vec2(rt_w, rt_h);
  vec2 tlPos = floor(cPos / vec2(size, size));
  tlPos *= size;
  int remX = int(mod(cPos.x, size));
  int remY = int(mod(cPos.y, size));
  if (remX == 0 && remY == 0)
    tlPos = cPos;
  vec2 blPos = tlPos;
  blPos.y += (size - 1.0);
  if ((remX == remY) ||
     (((int(cPos.x) - int(blPos.x)) == (int(blPos.y) - int(cPos.y)))))
  {
    if (invert == 1)
      c = vec4(0.2, 0.15, 0.05, 1.0);
    else
      c = texture2D(tex, tlPos * vec2(1.0/rt_w, 1.0/rt_h)) * 1.4;
  }
  else
  {
    if (invert == 1)
      c = texture2D(tex, tlPos * vec2(1.0/rt_w, 1.0/rt_h)) * 1.4;
    else
      c = vec4(0.0, 0.0, 0.0, 1.0);
  }
  return c;
}

void main (void)
{
  vec2 uv = gl_TexCoord[0].st;
  if (uv.y > 0.5)
  {
    gl_FragColor = PostFX(tex0, uv, time);
  }
  else
  {
    uv.y += 0.5;
    vec4 c1 = texture2D(tex0, uv);
    gl_FragColor = c1;
  }
}