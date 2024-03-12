//
// Simplex Noise
//

float plot(vec2 uv, float pct){
  return  smoothstep( pct - 0.01, pct, uv.y) -
          smoothstep( pct, pct + 0.01, uv.y);
}

float random(in float x) {
    return fract(sin(x) * 1e4);
}

float noise(in float x) {
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3. - 2. * f);
    return mix(random(i), random(i + 1.), u);
}

void main() {
    float y = noise(fragCoord.x * 3. + iTime);
    vec3 color = vec3(y);
    // float pct = plot(fragCoord, y);
    // color = (1. - pct) * color + pct * vec3(0.0, 1.0, 0.0);
    fragColor = vec4(color, 1.);
}
