#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <unordered_map>

namespace tgt {

class VulkanContext;
class Buffer;

// Indices into the per-material texture array (bindings 1-5 in the shader)
static constexpr int kPBRTexCount     = 5;
static constexpr int kPBRAlbedo       = 0;  // binding 1: _D
static constexpr int kPBRNormal       = 1;  // binding 2: _N
static constexpr int kPBRRoughness    = 2;  // binding 3: _R
static constexpr int kPBRMetallic     = 3;  // binding 4: _M
static constexpr int kPBRAO           = 4;  // binding 5: _AO
// IBL textures (bindings 6-8, shared across all submeshes)
static constexpr int kIBLTexCount     = 3;
static constexpr int kTotalTexBindings = kPBRTexCount + kIBLTexCount;  // 8

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

    // Allocate per-(submesh,frame) descriptor sets from the PBR pipeline's layout
    void createDescriptorResources(VkDescriptorSetLayout pbrDSLayout, int framesInFlight);

    // Upload view/proj/camera/light to the per-frame UBO for the given frame slot
    void updateUBO(int frameIdx, const float* view16, const float* proj16,
                   const float* cameraPos4, const float* lightDir4, const float* lightColor4);

    // Retrieve descriptor set for a specific (submesh, frame) pair
    VkDescriptorSet descriptorSet(int submeshIdx, int frameIdx) const;

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
        VkImage        image  = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView    view   = VK_NULL_HANDLE;
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

    // Flat list: index = submeshIdx * m_framesInFlight + frameIdx
    VkDescriptorPool             m_descPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_descSets;
    int m_framesInFlight = 0;
};

} // namespace tgt
