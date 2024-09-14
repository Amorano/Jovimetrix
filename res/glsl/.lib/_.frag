// name: BASIC FRAGMENT SHADER
// desc: draws 2 triangles as a quad for a surface to manipulate
// hide: true

uniform sampler2D image;

void mainImage( out vec4 fragColor, vec2 fragCoord ) {
  vec2 uv = fragCoord / iResolution.xy;
  fragColor = texture(image, uv);
}
