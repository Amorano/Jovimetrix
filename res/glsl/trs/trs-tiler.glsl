//
// Tile inputs
//

uniform vec2 uTile;

void main (void) {
    vec2 st = fract(fragCoord * uTile);
    fragColor = texture(iChannel0, st);
}