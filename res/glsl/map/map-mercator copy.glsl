//
// Remap to a Mercator Projection
//

#define TAU 6.28318530718
#define PI 3.14159265359
#define RAD 3.14159265 / 180.
#define RAT 1. / TAU

uniform bool flip;

void main() {
    float x = min(max(fragCoord.x, 0), 1) - 0.5;
    float y = 0.5 - min(max(fragCoord.y, 0), 1);
    y = 90 - (360 * atan(exp(0 - y * TAU)) / PI);
    x = 0.5 + (RAT * 360 * x * RAD);
    y = 0.5 - (RAT * y * RAD);
    if (flip) {
        fragColor = texture(iChannel0, vec2(y, x));
    } else {
        fragColor = texture(iChannel0, vec2(x, y));
    }
}



        float lat = (TexCoord.y - 0.5f) * PI;
        if(lat >= -1.48442222974871 && lat <= 1.48442222974871){
            float y = log(tan(PI4 + (lat / 2.0)));
            y = (y + PI) / TWO_PI;
            gl_FragColor = texture(ourTexture, vec2(TexCoord.x, y));
            return;
        }
        gl_FragColor = vec4(1, 1, 1, 0);