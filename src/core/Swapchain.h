#pragma once
#include <vulkan/vulkan.h>
#include <vector>

namespace tgt {

class VulkanContext;
class Window;

class Swapchain {
public:
    static constexpr int kMaxFramesInFlight = 2;

    Swapchain(VulkanContext& ctx, Window& window, VkSurfaceKHR surface);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    void recreate();

    VkResult acquireNextImage(uint32_t* imageIndex);
    VkResult submitAndPresent(uint32_t imageIndex, VkCommandBuffer cmd);

    VkSwapchainKHR           handle()          const { return m_swapchain; }
    VkFormat                 imageFormat()      const { return m_imageFormat; }
    VkExtent2D               extent()           const { return m_extent; }
    uint32_t                 imageCount()       const { return static_cast<uint32_t>(m_images.size()); }
    VkImageView              imageView(uint32_t i) const { return m_imageViews[i]; }
    VkImage                  image(uint32_t i)    const { return m_images[i]; }
    uint32_t                 currentFrame()     const { return m_currentFrame; }

private:
    void create();
    void cleanup();

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
    VkPresentModeKHR   choosePresentMode(const std::vector<VkPresentModeKHR>& modes);
    VkExtent2D         chooseExtent(const VkSurfaceCapabilitiesKHR& caps);

    VulkanContext&   m_ctx;
    Window&          m_window;
    VkSurfaceKHR     m_surface;

    VkSwapchainKHR          m_swapchain  = VK_NULL_HANDLE;
    std::vector<VkImage>    m_images;
    std::vector<VkImageView> m_imageViews;
    VkFormat                m_imageFormat{};
    VkExtent2D              m_extent{};

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence>     m_inFlightFences;
    std::vector<VkFence>     m_imagesInFlight;

    uint32_t m_currentFrame = 0;
};

} // namespace tgt
