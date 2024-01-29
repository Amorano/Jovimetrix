//
// Remap to a Mercator Projection
//

#define TAU 6.28318530718
#define PI  3.14159265359
#define PI4 0.78539816339f

uniform bool flip;

void main() {
    fragColor = vec4(1, 1, 1, 0);
    float lat = (fragCoord.y - 0.5f) * PI;
    if(lat >= -1.4975 && lat <= 1.4975) {
        float y = log(tan(PI4 + (lat / 2.0)));
        y = (y + PI) / TAU;
        if (flip) {
            fragColor = texture(iChannel0, vec2(y, fragCoord.x));
        } else {
            fragColor = texture(iChannel0, vec2(fragCoord.x, y));
        }
    }
}
