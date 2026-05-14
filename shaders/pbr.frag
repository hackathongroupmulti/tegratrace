#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_ray_query           : require

layout(set = 0, binding = 0) uniform PBRUniform {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor; // xyz = colour, w = intensity
} ubo;

// Bindless 2D texture heap: material maps + BRDF LUT at pc.brdfLutIdx
layout(set = 0, binding = 1) uniform sampler2D tex2D[];
// Fixed IBL cubemap bindings (shared across all submeshes)
layout(set = 0, binding = 2) uniform samplerCube texEnvPrefiltered; // GGX specular prefilter
layout(set = 0, binding = 3) uniform samplerCube texIrradiance;     // diffuse irradiance
// TLAS for inline ray queries (binding 4, partially bound — only valid when rtEnabled != 0)
layout(set = 0, binding = 4) uniform accelerationStructureEXT tlas;

// Push constants: mat4 model (vert) + 6 bindless texture indices + rtEnabled (frag)
layout(push_constant) uniform PC {
    mat4 model;
    uint albedoIdx;
    uint normalIdx;
    uint roughIdx;
    uint metallicIdx;
    uint aoIdx;
    uint brdfLutIdx;
    uint rtEnabled;
} pc;

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

void main() {
    // Sample material maps via bindless indices
    vec4 albedoSample = texture(tex2D[nonuniformEXT(pc.albedoIdx)],   fragUV);

    // Alpha test: discard near-transparent pixels (hair / ATOC meshes)
    if (albedoSample.a < 0.1) discard;

    float roughness = max(texture(tex2D[nonuniformEXT(pc.roughIdx)],    fragUV).r, 0.04);
    float metallic  = texture(tex2D[nonuniformEXT(pc.metallicIdx)],     fragUV).r;
    float ao        = texture(tex2D[nonuniformEXT(pc.aoIdx)],           fragUV).r;
    vec3  normalTS  = texture(tex2D[nonuniformEXT(pc.normalIdx)],       fragUV).rgb * 2.0 - 1.0;

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

    // Ray-traced hard shadow: shoot a shadow ray toward the directional light.
    // rtEnabled is pushed as 1 when the TLAS is built and bound; otherwise 0 (no shadow ray).
    float shadow = 1.0;
    if (pc.rtEnabled != 0u && NdotL > 0.0) {
        vec3 shadowOrigin = fragWorldPos + N * 0.005;  // small bias along surface normal
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, tlas,
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
            0xFF, shadowOrigin, 0.001, L, 200.0);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
            shadow = 0.08;  // retain ambient contribution even in shadow
    }

    vec3 Lo = (diffuse + specular) * lightRad * NdotL * shadow;

    // IBL split-sum ambient
    vec3 kS_ibl = F_Schlick(NdotV, F0);
    vec3 kD_ibl = (vec3(1.0) - kS_ibl) * (1.0 - metallic);

    vec3 irradiance = texture(texIrradiance, N).rgb;
    vec3 diffuseIBL = kD_ibl * irradiance * albedo;

    vec3 R = reflect(-V, N);
    vec3 prefSpec = textureLod(texEnvPrefiltered, R, roughness * 7.0).rgb;

    // BRDF LUT also lives in the bindless 2D heap
    vec2 brdfSample = texture(tex2D[nonuniformEXT(pc.brdfLutIdx)], vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefSpec * (kS_ibl * brdfSample.x + brdfSample.y);

    vec3 ambient = (diffuseIBL + specularIBL) * ao;

    vec3 color = ambient + Lo;

    // Reinhard tonemapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, albedoSample.a);
}
