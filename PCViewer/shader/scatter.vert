#version 450

layout(std430, binding = 0) buffer Dat{
	float d[];
} data;

layout(binding = 1, r8) uniform imageBuffer act; 

layout(binding = 2) uniform Buffer{
    float radius;
    uint showInactivePoints;
}ubo;

layout(push_constant) uniform PushConstants{
    uint posX;
    uint posY;
}pConst;

void main() {
	gl_VertexIndex;
}