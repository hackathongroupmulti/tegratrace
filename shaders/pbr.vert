#version 450

layout(binding = 0) uniform PBRUniform {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor; // xyz = colour, w = intensity
} ubo;

// Full push constant block shared with fragment shader (must match PBRDrawPC in Renderer.cpp)
layout(push_constant) uniform PC {
    mat4 model;
    uint albedoIdx;
    uint normalIdx;
    uint roughIdx;
    uint metallicIdx;
    uint aoIdx;
    uint brdfLutIdx;
    uint rtEnabled;   // 1 = shoot shadow ray via VK_KHR_ray_query, 0 = no shadow
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBitangent;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out mat3 fragTBN; // consumes locations 2, 3, 4

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    fragWorldPos  = worldPos.xyz;
    fragUV        = inTexCoord;

    mat3 normalMat = transpose(inverse(mat3(pc.model)));
    vec3 N = normalize(normalMat * inNormal);
    vec3 T = normalize(normalMat * inTangent);
    // Re-orthogonalise T against N (Gram-Schmidt)
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    fragTBN = mat3(T, B, N);

    gl_Position = ubo.proj * ubo.view * worldPos;
}
