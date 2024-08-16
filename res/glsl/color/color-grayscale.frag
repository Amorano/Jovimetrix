// name: GRAYSCALE
// desc: Convert input to grayscale
// category: COLOR

// default grayscale using NTSC conversion weights
uniform sampler2D image; //                                 | MASK, RGB or RGBA
uniform vec3 convert;    // 0.299, 0.587, 0.114; 0; 1; 0.01 | Scalar for each channel

void mainImage( out vec4 fragColor, vec2 fragCoord ) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 color = texture(image, uv);
    vec3 gray = vec3(dot(color.rgb, convert));
    fragColor = vec4(gray, color.a);
}