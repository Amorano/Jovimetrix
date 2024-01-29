uniform sampler2D sceneTex; // 0
uniform float vx_offset;
uniform float rt_w; // GeeXLab built-in
uniform float rt_h; // GeeXLab built-in
uniform float pixel_w; // 15.0
uniform float pixel_h; // 10.0
void main()
{
  vec2 uv = gl_TexCoord[0].xy;

  vec3 tc = vec3(1.0, 0.0, 0.0);
  if (uv.x < (vx_offset-0.005))
  {
    float dx = pixel_w*(1./rt_w);
    float dy = pixel_h*(1./rt_h);
    vec2 coord = vec2(dx*floor(uv.x/dx),
                      dy*floor(uv.y/dy));
    tc = texture2D(sceneTex, coord).rgb;
  }
  else if (uv.x>=(vx_offset+0.005))
  {
    tc = texture2D(sceneTex, uv).rgb;
  }
	gl_FragColor = vec4(tc, 1.0);
}