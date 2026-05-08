#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <array>

namespace tgt {

class VulkanContext;
class Buffer;

// Indices into the per-material texture array
static constexpr int kPBRTexCount     = 5;
static constexpr int kPBRAlbedo       = 0;  // _D
static constexpr int kPBRNormal       = 1;  // _N
static constexpr int kPBRRoughness    = 2;  // _R
static constexpr int kPBRMetallic     = 3;  // _M
static constexpr int kPBRAO           = 4;  // _AO

// Matches the UBO layout in pbr.vert / pbr.frag
struct PBRUBOData {
    float view[16];       // offset   0
    float proj[16];       // offset  64
    float cameraPos[4];   // offset 128
    float lightDir[4];    // offset 144
    float lightColor[4];  // offset 160  (xyz=colour, w=intensity)
};                        // total  176 bytes

struct PBRMaterial {
    std::string    name;
    VkImage        images[kPBRTexCount]    = {};
    VkDeviceMemory memories[kPBRTexCount]  = {};
    VkImageView    views[kPBRTexCount]     = {};
    bool           hasMap[kPBRTexCount]    = {};
};

struct PBRSubmesh {
    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    uint32_t    indexCount  = 0;
    int         materialIdx = -1;
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

private:
    // Load a texture from disk, upload to GPU; returns VK_NULL_HANDLE on failure
    VkImageView loadTextureFile(const std::string& path, bool srgb,
                                VkImage& outImage, VkDeviceMemory& outMem,
                                VkCommandPool pool);

    // Create 1×1 fallback textures (white and flat-normal)
    void createFallbackTextures(VkCommandPool pool);

    // Try several path combinations relative to the FBX directory
    std::string resolveTexPath(const std::string& rawPath) const;

    void destroyMaterials();

    VulkanContext& m_ctx;
    std::string    m_dir;  // directory of the loaded FBX

    std::vector<PBRSubmesh>  m_submeshes;
    std::vector<PBRMaterial> m_materials;

    // Fallback: solid white (for albedo/roughness/metallic/AO)
    VkImage        m_fbWhiteImg = VK_NULL_HANDLE;
    VkDeviceMemory m_fbWhiteMem = VK_NULL_HANDLE;
    VkImageView    m_fbWhiteView = VK_NULL_HANDLE;
    // Fallback: flat normal {128,128,255} → (0,0,1) in tangent space
    VkImage        m_fbNormImg  = VK_NULL_HANDLE;
    VkDeviceMemory m_fbNormMem  = VK_NULL_HANDLE;
    VkImageView    m_fbNormView = VK_NULL_HANDLE;

    VkSampler m_sampler = VK_NULL_HANDLE;

    // Per-frame UBO buffers
    std::vector<std::unique_ptr<Buffer>> m_uboBuffers;

    // Flat list: index = submeshIdx * m_framesInFlight + frameIdx
    VkDescriptorPool             m_descPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_descSets;
    int m_framesInFlight = 0;
};

} // namespace tgt
