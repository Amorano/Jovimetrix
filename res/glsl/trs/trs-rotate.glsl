//
// Rotate input
//

uniform float angle;
uniform vec2 center;

void main()
{
    float rads = radians(angle);
    float s = sin(rads);
    float c = cos(rads);

    vec2 rotated = fragCoord - center;
    float rotatedX = rotated.x * c - rotated.y * s;
    float rotatedY = rotated.x * s + rotated.y * c;
    rotated = vec2(rotatedX, rotatedY) + center;
    fragColor = texture(iChannel0, rotated);
}