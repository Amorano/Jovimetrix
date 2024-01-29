//

uniform vec3 target_tone = vec3(1.2, 1.0, 0.8);
uniform float opacity = 0.75;

// NTSC standard grey
// uniform vec3 conversion = vec3(0.299, 0.587, 0.114);

void main() {
    vec3 color = texture2D(iChannel0, fragCoord);
    // float gray = dot(color.rgb, conversion);
	vec3 target = color * target_tone;
	fragColor = mix(color.rgb, target, opacity);
}