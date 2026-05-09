#version 450

layout(binding = 0) uniform PBRUniform {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor; // xyz = colour, w = intensity
} ubo;

// Per-material texture maps (bindings 1-5)
layout(binding = 1) uniform sampler2D texAlbedo;
layout(binding = 2) uniform sampler2D texNormal;
layout(binding = 3) uniform sampler2D texRoughness;
layout(binding = 4) uniform sampler2D texMetallic;
layout(binding = 5) uniform sampler2D texAO;
// IBL resources (bindings 6-7, shared across all submeshes)
layout(binding = 6) uniform sampler2D texEnv;     // equirectangular environment map
layout(binding = 7) uniform sampler2D texBrdfLut; // GGX split-sum LUT: (NdotV, roughness) -> (scale, bias)

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in mat3 fragTBN; // occupies locations 2, 3, 4

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

// Smith combined geometry function
float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

// Fresnel-Schlick approximation
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Equirectangular UV from direction vector
vec2 equirUV(vec3 dir) {
    return vec2(atan(dir.z, dir.x) / (2.0 * PI) + 0.5,
                acos(clamp(dir.y, -1.0, 1.0)) / PI);
}

void main() {
    vec4 albedoSample = texture(texAlbedo, fragUV);

    // Alpha test: discard near-transparent pixels (hair / ATOC meshes)
    if (albedoSample.a < 0.1) discard;

    float roughness = max(texture(texRoughness, fragUV).r, 0.04);
    float metallic  = texture(texMetallic,  fragUV).r;
    float ao        = texture(texAO,        fragUV).r;
    vec3  normalTS  = texture(texNormal,    fragUV).rgb * 2.0 - 1.0;

    vec3 albedo = albedoSample.rgb;

    vec3 N = normalize(fragTBN * normalTS);
    vec3 V = normalize(ubo.cameraPos.xyz - fragWorldPos);
    vec3 L = normalize(ubo.lightDir.xyz);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Cook-Torrance specular BRDF (direct light)
    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3  F = F_Schlick(HdotV, F0);
    vec3  specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    vec3 lightRad = ubo.lightColor.rgb * ubo.lightColor.w;
    vec3 Lo = (diffuse + specular) * lightRad * NdotL;

    // IBL split-sum ambient (equirectangular env map + precomputed BRDF LUT)
    vec3 kS_ibl = F_Schlick(NdotV, F0);
    vec3 kD_ibl = (vec3(1.0) - kS_ibl) * (1.0 - metallic);

    // Diffuse IBL: sample env map at N with high LOD to approximate irradiance convolution
    vec3 irradiance = textureLod(texEnv, equirUV(N), 5.0).rgb;
    vec3 diffuseIBL = kD_ibl * irradiance * albedo;

    // Specular IBL: sample env map at reflection direction, LOD controlled by roughness
    vec3 R = reflect(-V, N);
    vec3 prefSpec = textureLod(texEnv, equirUV(R), roughness * 6.0).rgb;

    // BRDF LUT encodes (scale, bias) for the Fresnel term in split-sum form
    vec2 brdfSample = texture(texBrdfLut, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefSpec * (kS_ibl * brdfSample.x + brdfSample.y);

    vec3 ambient = (diffuseIBL + specularIBL) * ao;

    vec3 color = ambient + Lo;

    // Reinhard tonemapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, albedoSample.a);
}
