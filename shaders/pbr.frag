#version 450

layout(binding = 0) uniform PBRUniform {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor; // xyz = colour, w = intensity
} ubo;

// Binding slots: 1=albedo(_D), 2=normal(_N), 3=roughness(_R), 4=metallic(_M), 5=AO
layout(binding = 1) uniform sampler2D texAlbedo;
layout(binding = 2) uniform sampler2D texNormal;
layout(binding = 3) uniform sampler2D texRoughness;
layout(binding = 4) uniform sampler2D texMetallic;
layout(binding = 5) uniform sampler2D texAO;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in mat3 fragTBN; // locations 2, 3, 4

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// GGX/Trowbridge-Reitz Normal Distribution Function
float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX single-sided geometry term
float G_SchlickGGX(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / (NdotX * (1.0 - k) + k);
}

// Smith's combined geometry function
float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

// Fresnel-Schlick approximation
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec4  albedoSample = texture(texAlbedo,    fragUV);

    // Alpha test: discard near-transparent pixels (handles hair / ATOC meshes)
    if (albedoSample.a < 0.1) discard;

    float roughness = max(texture(texRoughness, fragUV).r, 0.04);
    float metallic  = texture(texMetallic,  fragUV).r;
    float ao        = texture(texAO,        fragUV).r;
    vec3  normalTS  = texture(texNormal,    fragUV).rgb * 2.0 - 1.0;

    // Albedo from sRGB texture — already in linear space because the image was
    // created with VK_FORMAT_R8G8B8A8_SRGB which hardware converts on sample.
    vec3 albedo = albedoSample.rgb;

    // Normal in world space via TBN
    vec3 N = normalize(fragTBN * normalTS);
    vec3 V = normalize(ubo.cameraPos.xyz - fragWorldPos);
    vec3 L = normalize(ubo.lightDir.xyz);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // Dielectric F0 = 0.04; metals use albedo colour
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Cook-Torrance specular BRDF
    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3  F = F_Schlick(HdotV, F0);
    vec3  specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    // Lambertian diffuse (energy-conserving: metals have no diffuse term)
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    // Direct lighting
    vec3 lightRad = ubo.lightColor.rgb * ubo.lightColor.w;
    vec3 Lo = (diffuse + specular) * lightRad * NdotL;

    // Ambient: constant IBL approximation scaled by AO
    vec3 ambient = vec3(0.03) * albedo * ao;

    vec3 color = ambient + Lo;

    // Reinhard tonemapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, albedoSample.a);
}
