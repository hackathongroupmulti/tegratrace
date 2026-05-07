#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple Lambertian diffuse
    vec3  lightDir = normalize(vec3(1.0, 2.0, 1.5));
    float diffuse  = max(dot(normalize(fragNormal), lightDir), 0.0);
    vec3  ambient  = fragColor * 0.15;
    outColor = vec4(ambient + fragColor * diffuse, 1.0);
}
