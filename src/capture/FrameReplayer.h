#pragma once
#include <vulkan/vulkan.h>
#include <string>

namespace tgt {
class VulkanContext; class Swapchain; class RenderPass;
class Pipeline; class Renderer; class RegressionTester;

class FrameReplayer {
public:
    FrameReplayer(VulkanContext& ctx, Swapchain& swapchain, RenderPass& renderPass,
                  Pipeline& pipeline, Renderer& renderer, RegressionTester& regression);
    // Returns true if replay succeeded (and PSNR passed if reference exists)
    bool replay(const std::string& captureJsonPath, VkCommandPool cmdPool);
private:
    VulkanContext&    m_ctx;
    Swapchain&        m_swapchain;
    RenderPass&       m_renderPass;
    Pipeline&         m_pipeline;
    Renderer&         m_renderer;
    RegressionTester& m_regression;
};
} // namespace tgt
