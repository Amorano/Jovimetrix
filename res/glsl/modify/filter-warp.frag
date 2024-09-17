// name: DIRECTIONAL WARP
// desc: Domain warp an image with a direction and distortion map
// category: MODIFY

#include .lib/const.lib

uniform sampler2D image;      //          | RGB(A) image
uniform sampler2D distortion; //          | RGB(A) image used as a LUMA mask for distortion
uniform sampler2D direction;  //          | RGB(A) image used as a LUMA mask for direction
uniform float strength;       // 64;0;;1  | Pixel data range allowed

vec2 warp(vec2 uv)
{
    vec4 uv_distortion = texture(distortion, uv);
    float distortion_val = dot(uv_distortion.rgb, vec3(0.299, 0.587, 0.114));

    vec4 uv_direction = texture(direction, uv);
    float angle = dot(uv_direction.rgb, vec3(0.299, 0.587, 0.114)) * M_TAU;
    vec2 direction_val = vec2(cos(angle), sin(angle));
    uv += direction_val * distortion_val * strength / iResolution.xy;
    return uv;
 }

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = warp(fragCoord / iResolution.xy);
    fragColor = texture(image, uv);
}
