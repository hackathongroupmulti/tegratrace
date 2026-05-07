#pragma once
#include <vulkan/vulkan.h>
#include <vector>

namespace tgt {

class VulkanContext;
class Swapchain;

class RenderPass {
public:
    RenderPass(VulkanContext& ctx, Swapchain& swapchain);
    ~RenderPass();

    RenderPass(const RenderPass&) = delete;
    RenderPass& operator=(const RenderPass&) = delete;

    void recreateFramebuffers();

    VkRenderPass  handle()               const { return m_renderPass; }
    VkFramebuffer framebuffer(uint32_t i) const { return m_framebuffers[i]; }

private:
    void createDepthResources();
    void createFramebuffers();
    void cleanupFramebuffers();

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;

    VkRenderPass              m_renderPass   = VK_NULL_HANDLE;
    VkImage                   m_depthImage   = VK_NULL_HANDLE;
    VkDeviceMemory            m_depthMemory  = VK_NULL_HANDLE;
    VkImageView               m_depthView    = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_framebuffers;
    VkFormat                  m_depthFormat{};
};

} // namespace tgt
