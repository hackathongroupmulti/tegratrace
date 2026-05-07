#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    // Checkerboard pattern from UV for visual validation
    float checker = mod(floor(fragTexCoord.x * 8.0) + floor(fragTexCoord.y * 8.0), 2.0);
    vec3  pattern = mix(fragColor, fragColor * 0.6, checker);
    outColor = vec4(pattern, 1.0);
}
