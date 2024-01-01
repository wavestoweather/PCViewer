#version 450

layout(location = 0) out uint dataIndex;

void main() {
    dataIndex = gl_VertexIndex;
}