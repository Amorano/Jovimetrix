// name: BASIC VERTEX SHADER
// desc: draws 2 triangles as a quad for a surface to manipulate

#version 460
precision highp float;

void main()
{
    vec2 verts[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(verts[gl_VertexID], 0, 1);
}
