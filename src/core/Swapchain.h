#pragma once
#include <vulkan/vulkan.h>
#include <vector>

namespace tgt {

class VulkanContext;
class Window;

class Swapchain {
public:
    static constexpr int kMaxFramesInFlight = 2;

    // Pass window=nullptr and surface=VK_NULL_HANDLE for headless offscreen mode
    Swapchain(VulkanContext& ctx, Window* window, VkSurfaceKHR surface);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    void recreate();

    VkResult acquireNextImage(uint32_t* imageIndex);
    // cullSem: optional timeline semaphore that must be waited before DRAW_INDIRECT stage
    VkResult submitAndPresent(uint32_t imageIndex, VkCommandBuffer cmd,
                              VkSemaphore cullSem = VK_NULL_HANDLE, uint64_t cullWaitVal = 0);

    VkSwapchainKHR           handle()          const { return m_swapchain; }
    VkFormat                 imageFormat()      const { return m_imageFormat; }
    VkExtent2D               extent()           const { return m_extent; }
    uint32_t                 imageCount()       const { return static_cast<uint32_t>(m_images.size()); }
    VkImageView              imageView(uint32_t i) const { return m_imageViews[i]; }
    VkImage                  image(uint32_t i)    const { return m_images[i]; }
    uint32_t                 currentFrame()     const { return m_currentFrame; }

    bool isHeadless() const { return m_swapchain == VK_NULL_HANDLE; }
    // Layout the color attachment is in after the render pass completes
    VkImageLayout colorFinalLayout() const {
        return isHeadless() ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                            : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

private:
    void create();
    void createOffscreen();
    void cleanup();
    void cleanupOffscreen();
    VkResult acquireNextImageOffscreen(uint32_t* imageIndex);
    VkResult submitAndPresentOffscreen(uint32_t imageIndex, VkCommandBuffer cmd);

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
    VkPresentModeKHR   choosePresentMode(const std::vector<VkPresentModeKHR>& modes);
    VkExtent2D         chooseExtent(const VkSurfaceCapabilitiesKHR& caps);

    VulkanContext&   m_ctx;
    Window*          m_window;
    VkSurfaceKHR     m_surface;

    VkSwapchainKHR            m_swapchain  = VK_NULL_HANDLE;
    std::vector<VkImage>      m_images;
    std::vector<VkImageView>  m_imageViews;
    std::vector<VkDeviceMemory> m_imageMemories; // used by offscreen path only
    VkFormat                  m_imageFormat{};
    VkExtent2D                m_extent{};

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence>     m_inFlightFences;
    std::vector<VkFence>     m_imagesInFlight;

    uint32_t m_currentFrame = 0;
};

} // namespace tgt
