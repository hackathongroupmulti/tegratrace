#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <unordered_map>
#include "Buffer.h"

namespace tgt {

class VulkanContext;

// Indices into the per-material texture array (bindings 1-5 in the shader)
static constexpr int kPBRTexCount      = 5;
static constexpr int kPBRAlbedo        = 0;
static constexpr int kPBRNormal        = 1;
static constexpr int kPBRRoughness     = 2;
static constexpr int kPBRMetallic      = 3;
static constexpr int kPBRAO            = 4;
// IBL textures (classic layout bindings 6-8)
static constexpr int kIBLTexCount      = 3;
static constexpr int kTotalTexBindings = kPBRTexCount + kIBLTexCount;  // 8 (classic layout)
// Bindless: max descriptors in the sampler2D[] array
static constexpr uint32_t kMaxBindlessTextures = 256;

// Meshlet: packed geometry for VK_EXT_mesh_shader task/mesh pipeline (max 64 verts, 124 tris)
struct Meshlet {
    uint32_t vertexOffset;    // first index into meshlet vertex buffer
    uint32_t triangleOffset;  // first index into meshlet triangle buffer (3 per tri)
    uint32_t vertexCount;     // <= 64
    uint32_t triangleCount;   // <= 124
    float    sphere[4];       // xyz = center in model space, w = radius
};

// Per-submesh bindless texture indices pushed via push constants
struct TexIndices {
    uint32_t albedoIdx   = 0;
    uint32_t normalIdx   = 0;
    uint32_t roughIdx    = 0;
    uint32_t metallicIdx = 0;
    uint32_t aoIdx       = 0;
    uint32_t brdfLutIdx  = 0;
};

// Matches the UBO layout in pbr.vert / pbr.frag
struct PBRUBOData {
    float view[16];       // offset   0
    float proj[16];       // offset  64
    float cameraPos[4];   // offset 128
    float lightDir[4];    // offset 144
    float lightColor[4];  // offset 160  (xyz=colour, w=intensity)
};                        // total  176 bytes

struct PBRMaterial {
    std::string name;
    VkImageView views[kPBRTexCount]  = {};   // borrowed from PBRModel::m_texCache
    bool        hasMap[kPBRTexCount] = {};
};

// GPU-driven: stores offsets into consolidated VBO/IBO instead of owning buffers
struct PBRSubmesh {
    uint32_t    indexCount    = 0;
    uint32_t    firstIndex    = 0;   // byte offset into consolidated IBO (in indices)
    int32_t     vertexOffset  = 0;   // vertex index added to each index (into consolidated VBO)
    int         materialIdx   = -1;
    std::string name;
};

class PBRModel {
public:
    explicit PBRModel(VulkanContext& ctx);
    ~PBRModel();

    PBRModel(const PBRModel&) = delete;
    PBRModel& operator=(const PBRModel&) = delete;

    // Load FBX via Assimp; must be called before createDescriptorResources
    bool load(const std::string& fbxPath, VkCommandPool pool);

    // Allocate descriptor sets from the PBR pipeline's layout.
    // bindless=true: one set per frame with sampler2D[] array; false: one set per (submesh×frame)
    void createDescriptorResources(VkDescriptorSetLayout pbrDSLayout, int framesInFlight,
                                   bool bindless = false);

    // Ray tracing: build BLAS/TLAS from consolidated geometry; call after load().
    // modelMat16 = column-major float[16] transform applied to the TLAS instance.
    void buildAccelerationStructures(VkCommandPool pool, const float* modelMat16);
    void writeTLASDescriptors();     // update binding 4 in all desc sets (PBR + mesh) after TLAS build
    bool rtBuilt() const { return m_rtBuilt; }

    // Mesh shaders: create per-frame descriptor sets for the mesh pipeline layout (bindings 0-8).
    // Must be called after createDescriptorResources() (reuses bindless index assignments).
    void createMeshDescriptorResources(VkDescriptorSetLayout meshLayout, int framesInFlight);
    VkDescriptorSet meshDescriptorSet(int frameIdx) const;
    uint32_t meshletCount()                    const { return static_cast<uint32_t>(m_meshlets.size()); }
    uint32_t submeshMeshletOffset(int s)       const { return m_submeshMeshletOffsets[static_cast<size_t>(s)]; }
    uint32_t submeshMeshletCount(int s)        const { return m_submeshMeshletCounts[static_cast<size_t>(s)]; }

    // Upload view/proj/camera/light to the per-frame UBO for the given frame slot
    void updateUBO(int frameIdx, const float* view16, const float* proj16,
                   const float* cameraPos4, const float* lightDir4, const float* lightColor4);

    // Retrieve descriptor set: with bindless, returns the frame-level set (submeshIdx ignored)
    VkDescriptorSet descriptorSet(int submeshIdx, int frameIdx) const;

    // Bindless: per-submesh texture indices for push constants
    const TexIndices& texIndices(int submeshIdx) const;
    bool              isBindless() const { return m_bindless; }

    const std::vector<PBRSubmesh>& submeshes() const { return m_submeshes; }
    bool isLoaded() const { return !m_submeshes.empty(); }

    // GPU-driven indirect: consolidated geometry buffers + indirect draw commands
    VkBuffer consolidatedVBO()   const;
    VkBuffer consolidatedIBO()   const;
    VkBuffer indirectBuffer()    const;
    VkBuffer sphereBuffer()      const;  // per-submesh world-space bounding spheres (SSBO)
    uint32_t drawCount()         const { return static_cast<uint32_t>(m_submeshes.size()); }

private:
    // Load a texture from disk, upload to GPU with full mip chain; returns VK_NULL_HANDLE on failure
    VkImageView loadTextureFile(const std::string& path, bool srgb,
                                VkImage& outImage, VkDeviceMemory& outMem,
                                VkCommandPool pool);

    // Create 1×1 fallback textures (white and flat-normal)
    void createFallbackTextures(VkCommandPool pool);

    // Try several path combinations relative to the FBX directory
    std::string resolveTexPath(const std::string& rawPath) const;

    void destroyMaterials();
    void destroyTextureCache();

    // IBL: generate the GGX BRDF integration LUT (CPU) and produce cubemap resources from m_dir
    void generateBRDFLut(VkCommandPool pool);
    void loadEnvMap(VkCommandPool pool);       // loads equirect HDR into m_envEquirectImg
    void buildIBLCubemaps(VkCommandPool pool); // runs env_to_cube, prefilter, irradiance passes

    // Texture cache: resolved absolute path → GPU objects (shared across materials)
    struct CachedTexture {
        VkImage        image       = VK_NULL_HANDLE;
        VkDeviceMemory memory      = VK_NULL_HANDLE;
        VkImageView    view        = VK_NULL_HANDLE;
        uint32_t       bindlessIdx = 0;  // index into tex2D[] when bindless
    };
    std::unordered_map<std::string, CachedTexture> m_texCache;

    VulkanContext& m_ctx;
    std::string    m_dir;  // directory of the loaded FBX

    std::vector<PBRSubmesh>  m_submeshes;
    std::vector<PBRMaterial> m_materials;

    // Fallback: solid white (for albedo/roughness/metallic/AO)
    VkImage        m_fbWhiteImg  = VK_NULL_HANDLE;
    VkDeviceMemory m_fbWhiteMem  = VK_NULL_HANDLE;
    VkImageView    m_fbWhiteView = VK_NULL_HANDLE;
    // Fallback: flat normal {128,128,255} → (0,0,1) in tangent space
    VkImage        m_fbNormImg   = VK_NULL_HANDLE;
    VkDeviceMemory m_fbNormMem   = VK_NULL_HANDLE;
    VkImageView    m_fbNormView  = VK_NULL_HANDLE;

    // IBL: equirectangular HDR source (intermediate; not bound to shader)
    VkImage        m_envEquirectImg  = VK_NULL_HANDLE;
    VkDeviceMemory m_envEquirectMem  = VK_NULL_HANDLE;
    VkImageView    m_envEquirectView = VK_NULL_HANDLE;
    // IBL: source cubemap converted from equirect (input to prefilter + irradiance passes)
    VkImage        m_envCubeImg  = VK_NULL_HANDLE;
    VkDeviceMemory m_envCubeMem  = VK_NULL_HANDLE;
    VkImageView    m_envCubeView = VK_NULL_HANDLE; // full array view (samplerCube input)
    // IBL: GGX specular prefiltered cubemap (binding 6)
    VkImage        m_prefilterImg  = VK_NULL_HANDLE;
    VkDeviceMemory m_prefilterMem  = VK_NULL_HANDLE;
    VkImageView    m_prefilterView = VK_NULL_HANDLE;
    std::vector<VkImageView> m_prefilterMipViews;  // per-mip storage views for compute writes
    // IBL: GGX BRDF integration LUT (binding 7)
    VkImage        m_brdfLutImg  = VK_NULL_HANDLE;
    VkDeviceMemory m_brdfLutMem  = VK_NULL_HANDLE;
    VkImageView    m_brdfLutView = VK_NULL_HANDLE;
    // IBL: diffuse irradiance cubemap (binding 8)
    VkImage        m_irradianceImg  = VK_NULL_HANDLE;
    VkDeviceMemory m_irradianceMem  = VK_NULL_HANDLE;
    VkImageView    m_irradianceView = VK_NULL_HANDLE;

    VkSampler m_sampler = VK_NULL_HANDLE;

    // GPU-driven indirect: single VBO/IBO for all submeshes + VkDrawIndexedIndirectCommand array
    std::unique_ptr<Buffer> m_consolidatedVBO;
    std::unique_ptr<Buffer> m_consolidatedIBO;
    std::unique_ptr<Buffer> m_indirectBuffer;  // source (also STORAGE for cull CS read)
    std::unique_ptr<Buffer> m_sphereBuffer;    // per-submesh world-space bounding spheres

    // Per-frame UBO buffers
    std::vector<std::unique_ptr<Buffer>> m_uboBuffers;

    // Flat list: index = submeshIdx * m_framesInFlight + frameIdx (classic) or frameIdx (bindless)
    VkDescriptorPool             m_descPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_descSets;
    int m_framesInFlight = 0;

    // Bindless path: one TexIndices per submesh, m_bindless flag
    bool                     m_bindless    = false;
    std::vector<TexIndices>  m_texIndices;

    // Mesh shader meshlet data (built during load() when meshShaderSupported)
    void buildMeshletsInternal(const std::vector<PBRVertex>& verts,
                               const std::vector<uint32_t>& indices,
                               VkCommandPool pool);
    std::vector<Meshlet>   m_meshlets;
    std::vector<uint32_t>  m_meshletVerts;  // vertex indices into consolidated VBO
    std::vector<uint32_t>  m_meshletTris;   // local vertex indices per triangle (3 per tri)
    std::vector<uint32_t>  m_submeshMeshletOffsets;
    std::vector<uint32_t>  m_submeshMeshletCounts;
    std::unique_ptr<Buffer> m_meshletBuf;
    std::unique_ptr<Buffer> m_meshletVertBuf;
    std::unique_ptr<Buffer> m_meshletTriBuf;

    // Descriptor resources for mesh shader pipeline (separate from m_descPool/m_descSets)
    VkDescriptorPool             m_meshDescPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_meshDescSets;

    // Ray tracing acceleration structures
    VkAccelerationStructureKHR m_blasHandle = VK_NULL_HANDLE;
    std::unique_ptr<Buffer>    m_blasBuffer;
    VkDeviceAddress            m_blasDevAddr = 0;
    VkAccelerationStructureKHR m_tlasHandle = VK_NULL_HANDLE;
    std::unique_ptr<Buffer>    m_tlasBuffer;
    std::unique_ptr<Buffer>    m_tlasInstanceBuf;
    bool                       m_rtBuilt = false;
};

} // namespace tgt
