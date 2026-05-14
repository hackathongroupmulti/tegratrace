#pragma once
#include <vulkan/vulkan.h>
#include <string>

namespace tgt {

class VulkanContext;
class RenderPass;

struct PipelineConfig {
    std::string vertSpvPath;
    std::string fragSpvPath;
    std::string taskSpvPath;  // task shader (only when useMeshShader)
    std::string meshSpvPath;  // mesh shader (only when useMeshShader)
    bool        depthTest    = true;
    bool        depthWrite   = true;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkPolygonMode   polyMode = VK_POLYGON_MODE_FILL;
    bool        enableStats  = true;
    // Number of combined-image-sampler bindings (slots 1..textureBindingCount) — ignored when useBindless
    uint32_t    textureBindingCount = 1;
    // Use PBRVertex input layout instead of the default Vertex layout
    bool        usePBRVertex = false;
    // Bindless descriptor layout: binding 1 = sampler2D[], bindings 2-3 = samplerCube (IBL)
    // Push constants extended to 92 bytes (mat4 + 6 uint texIndices + rtEnabled, VERTEX|FRAGMENT)
    bool        useBindless  = false;
    // Mesh shader pipeline: task+mesh+frag, no vertex input state, adds SSBO bindings 5-8
    // Push constants extended to 100 bytes (adds meshletOffset + meshletCount for task/mesh)
    bool        useMeshShader = false;
};

class Pipeline {
public:
    Pipeline(VulkanContext& ctx, RenderPass& renderPass,
             VkExtent2D extent, const PipelineConfig& cfg);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    VkPipeline            handle()        const { return m_pipeline; }
    VkPipelineLayout      layout()        const { return m_layout; }
    VkDescriptorSetLayout dsLayout()      const { return m_dsLayout; }
    const std::string& name()            const { return m_name; }
    const std::string& vertSpvPath()     const { return m_vertSpvPath; }
    const std::string& fragSpvPath()     const { return m_fragSpvPath; }

    bool bindless() const { return m_cfg.useBindless; }

    // Reload SPIR-V from disk and recreate the VkPipeline in-place.
    // Returns true on success; keeps old pipeline on failure.
    bool tryReload();

private:
    VkPipeline buildGraphicsPipeline() const;  // uses stored m_cfg / m_renderPassHandle / m_extent

    VulkanContext& m_ctx;
    VkPipeline       m_pipeline        = VK_NULL_HANDLE;
    VkPipelineLayout m_layout          = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_dsLayout   = VK_NULL_HANDLE;
    std::string      m_name;
    std::string      m_vertSpvPath;
    std::string      m_fragSpvPath;

    // Stored for hot-reload
    VkRenderPass   m_renderPassHandle = VK_NULL_HANDLE;
    VkExtent2D     m_extent{};
    PipelineConfig m_cfg;
};

} // namespace tgt
