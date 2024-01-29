uniform sampler2D sceneTex; // 0
uniform float vx_offset;
uniform float rt_w; // GeeXLab built-in
uniform float rt_h; // GeeXLab built-in
uniform float hatch_y_offset; // 5.0
uniform float lum_threshold_1; // 1.0
uniform float lum_threshold_2; // 0.7
uniform float lum_threshold_3; // 0.5
uniform float lum_threshold_4; // 0.3
void main()
{
  vec2 uv = gl_TexCoord[0].xy;

  vec3 tc = vec3(1.0, 0.0, 0.0);
  if (uv.x < (vx_offset-0.005))
  {
    float lum = length(texture2D(sceneTex, uv).rgb);
    tc = vec3(1.0, 1.0, 1.0);

    if (lum < lum_threshold_1)
    {
      if (mod(gl_FragCoord.x + gl_FragCoord.y, 10.0) == 0.0)
        tc = vec3(0.0, 0.0, 0.0);
    }

    if (lum < lum_threshold_2)
    {
      if (mod(gl_FragCoord.x - gl_FragCoord.y, 10.0) == 0.0)
        tc = vec3(0.0, 0.0, 0.0);
    }

    if (lum < lum_threshold_3)
    {
      if (mod(gl_FragCoord.x + gl_FragCoord.y - hatch_y_offset, 10.0) == 0.0)
        tc = vec3(0.0, 0.0, 0.0);
    }

    if (lum < lum_threshold_4)
    {
      if (mod(gl_FragCoord.x - gl_FragCoord.y - hatch_y_offset, 10.0) == 0.0)
        tc = vec3(0.0, 0.0, 0.0);
    }
  }
  else if (uv.x>=(vx_offset+0.005))
  {
    tc = texture2D(sceneTex, uv).rgb;
  }

  gl_FragColor = vec4(tc, 1.0);
}