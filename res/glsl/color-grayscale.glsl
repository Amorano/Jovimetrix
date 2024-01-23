// default grayscale using NTSC conversion weights
uniform vec3 conversion; // (0.299, 0.587, 0.114)
void main() {
    vec4 color = texture2D(iChannel0, iUV);
    vec3 gray = vec3(dot(color.rgb, conversion));
    fragColor = vec4(gray, 1.0);
}