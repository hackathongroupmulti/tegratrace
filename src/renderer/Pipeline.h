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
};

class Pipeline {
public:
    Pipeline(VulkanContext& ctx, RenderPass& renderPass,
             VkExtent2D extent, const PipelineConfig& cfg);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    VkPipeline       handle()       const { return m_pipeline; }
    VkPipelineLayout layout()       const { return m_layout; }
    const std::string& name()       const { return m_name; }

private:
    VulkanContext& m_ctx;
    VkPipeline       m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_layout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_dsLayout = VK_NULL_HANDLE;
    std::string      m_name;
};

} // namespace tgt
