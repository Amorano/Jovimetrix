//
// Range select clipped at start and end
//

uniform vec3 start;
uniform vec3 end;

void main() {
    vec3 color = texture(iChannel0, fragCoord).xyz;
    if (color.r >= start.r && color.r <= end.r && color.g >= start.g && color.g <= end.g && color.b >= start.b && color.b <= end.b) {
        fragColor = vec4(1.0);
    } else {
        fragColor = vec4(vec3(0.0), 1.0);
    }
}