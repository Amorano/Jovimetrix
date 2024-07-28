// name: GRAYSCALE
// desc: Convert input to grayscale
//

// default grayscale using NTSC conversion weights
uniform sampler2D image;
uniform vec3 conversion; // 0.299, 0.587, 0.114;0;1;0.01 | Scalar for each channel

void mainImage( out vec4 fragColor, vec2 fragCoord ) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 color = texture(image, uv);
    vec3 gray = vec3(dot(color.rgb, conversion));
    fragColor = vec4(gray, color.a);
}