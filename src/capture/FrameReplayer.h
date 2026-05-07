#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include "validation/RegressionTester.h"

namespace tgt {
class VulkanContext; class Swapchain; class RenderPass;
class Pipeline; class Renderer;

class FrameReplayer {
public:
    FrameReplayer(VulkanContext& ctx, Swapchain& swapchain, RenderPass& renderPass,
                  Pipeline& pipeline, Renderer& renderer, RegressionTester& regression);
    // Returns the image diff result (psnrDb/passed valid if a reference PNG exists)
    ImageDiffResult replay(const std::string& captureJsonPath, VkCommandPool cmdPool);
private:
    VulkanContext&    m_ctx;
    Swapchain&        m_swapchain;
    RenderPass&       m_renderPass;
    Pipeline&         m_pipeline;
    Renderer&         m_renderer;
    RegressionTester& m_regression;
};
} // namespace tgt
