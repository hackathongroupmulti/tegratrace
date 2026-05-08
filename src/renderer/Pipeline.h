#pragma once
#include <vulkan/vulkan.h>
#include <string>

namespace tgt {

class VulkanContext;
class RenderPass;

struct PipelineConfig {
    std::string vertSpvPath;
    std::string fragSpvPath;
    bool        depthTest    = true;
    bool        depthWrite   = true;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkPolygonMode   polyMode = VK_POLYGON_MODE_FILL;
    bool        enableStats  = true;
    // Number of combined-image-sampler bindings (slots 1..textureBindingCount)
    uint32_t    textureBindingCount = 1;
    // Use PBRVertex input layout instead of the default Vertex layout
    bool        usePBRVertex = false;
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

private:
    VulkanContext& m_ctx;
    VkPipeline       m_pipeline  = VK_NULL_HANDLE;
    VkPipelineLayout m_layout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_dsLayout = VK_NULL_HANDLE;
    std::string      m_name;
    std::string      m_vertSpvPath;
    std::string      m_fragSpvPath;
};

} // namespace tgt
