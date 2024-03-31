//
// Range select clipped at start and end
//

uniform vec3 start;
uniform vec3 end;

void main() {
    vec4 color = texture(iChannel0, fragCoord);
    if (color.r >= start.r && color.r <= end.r && color.g >= start.g && color.g <= end.g && color.b >= start.b && color.b <= end.b) {
        fragColor = vec4(vec3(1.0), color.w);
    } else {
        fragColor = vec4(vec3(0.0), color.w);
    }
}