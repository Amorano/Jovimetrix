uniform sampler2D imageA;
uniform sampler2D imageB;

void mainImage( out vec4 fragColor, vec2 fragCoord ) {
  vec2 uv = fragCoord.xy / iResolution.xy;
  vec3 col = texture2D(imageA, uv).rgb;
  vec3 col2 = texture2D(imageB, uv).rgb;
  fragColor = vec4(mix(col, col2, 0.5), 1.0);
}