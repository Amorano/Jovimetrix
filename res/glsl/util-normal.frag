// name: NORMAL
// desc: Convert input into a Normal map
//

uniform sampler2D image; //      | Input image to convert into a normal map
uniform float scalar;    // 0.25 | Intensity of depth

const mat3 scharr_x = mat3(
    3.0, 10.0, 3.0,
    0.0, 0.0, 0.0,
    -3.0, -10.0, -3.0
);

const mat3 scharr_y = mat3(
    3.0, 0.0, -3.0,
    10.0, 0.0, -10.0,
    3.0, 0.0, -3.0
);

vec3 scharr(sampler2D tex, vec2 uv) {
    vec3 result = vec3(0.0);
    vec2 texelSize = 1.0 / iResolution.xy;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec3 color = texture(tex, uv + offset).rgb;
            float luminance = dot(color, vec3(0.299, 0.587, 0.114));
            result.x += luminance * scharr_x[i+1][j+1];
            result.y += luminance * scharr_y[i+1][j+1];
        }
    }
    return result;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    vec3 normal;
    normal.xy = scharr(image, uv).yx * scalar;
    normal.x *= -1.0;
    normal.z = 1.0;
    normal = normalize(normal);
    fragColor = vec4(normal * 0.5 + 0.5, 1.0);
}