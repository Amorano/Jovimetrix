// name: NORMAL
// desc: Convert input into a Normal map
// category: CREATE

uniform sampler2D image; //        | Input image to convert into a normal map
uniform float scalar;    // 1.00;0 | Intensity of base normal
uniform float detail;    // 1.00;0 | Intensity of detail normal
uniform bool flip;       //        | Reverse the Normal direction

const mat3 scharr_x = mat3(
     1.0,    10.0/3.0,  1.0,
     0.0,     0.0,      0.0,
    -1.0,   -10.0/3.0, -1.0
);

const mat3 scharr_y = mat3(
     1.0,     0.0,  -1.0,
    10.0/3.0, 0.0,  -10.0/3.0,
     1.0,     0.0,  -1.0
);

vec3 scharr(vec2 uv) {
    vec3 result = vec3(0.0);
    vec2 texelSize = 1.0 / iResolution.xy;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec3 color = texture(image, uv + offset).rgb;
            float luminance = dot(color, vec3(0.299, 0.587, 0.114));
            result.x += luminance * scharr_x[i+1][j+1] * detail;
            result.y += luminance * scharr_y[i+1][j+1] * detail;
        }
    }
    return result;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;

    // detailed normal
    vec3 normal = vec3(0.0, 0.0, 1.0);
    normal.xy = scharr(uv).xy * scalar;
    if (flip) {
        normal.xy = normal.yx;
    }
    normal.x *= -scalar;
    normal = normalize(normal) * 0.5 + 0.5;
    fragColor = vec4(normal, 1.0);
}