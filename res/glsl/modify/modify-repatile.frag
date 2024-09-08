// name: REPATILE
// desc: Generate a square grid
// category: MODIFY

uniform vec2 grid_xy;       // 16,16;0;512;1 | grid squares per width x height

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord / iResolution.xy;
    vec2 grid_pixel_space = iResolution.xy / grid_xy;
	vec2 grid_uv = fract(uv / grid_pixel_space) * grid_pixel_space;
	vec2 line = step(grid_uv, vec2(1.0));
	float val = max(line.x, line.y);
	fragColor = vec4(vec3(val), 1.0);
}