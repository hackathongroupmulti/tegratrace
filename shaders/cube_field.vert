#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    int col = gl_InstanceIndex % 8;
    int row = gl_InstanceIndex / 8;

    // 8x8 grid centered at origin, spaced 2.5 units apart, placed at z=-20
    vec3 offset = vec3(
        (float(col) - 3.5) * 2.5,
        (float(row) - 3.5) * 2.5,
        -20.0
    );

    // Apply model's rotation component only (upper-left 3x3 = no translation)
    vec3 rotated  = mat3(ubo.model) * inPosition;
    vec4 worldPos = vec4(rotated + offset, 1.0);

    gl_Position = ubo.proj * ubo.view * worldPos;

    // Per-instance color: vary by column and row
    float cu = float(col) / 7.0;
    float cv = float(row) / 7.0;
    fragColor    = inColor * vec3(0.2 + 0.8 * cu, 0.2 + 0.8 * cv, 1.0 - 0.5 * cu);
    fragTexCoord = inTexCoord;
}
